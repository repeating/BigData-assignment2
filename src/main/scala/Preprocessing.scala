import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{Dataset, Row, SparkSession}

// object we used for choosing classification model and feature extraction algorithm
object Preprocessing {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
    val spark = SparkSession.builder()
      .appName("Spark sess")
      .master("local[*]")
      .getOrCreate()

    // define colomns of input data
    val schema = StructType(
      Array(StructField("Id", DoubleType),
        StructField("class", DoubleType),
        StructField("text", StringType),
      )
    )
    // reading dataset to dataframe object
    val df_raw = spark
      .read
      .option("header", "true")
      .schema(schema)
      .csv("dataset")
    // preprocess data so it clean tokens
    val df_clean = Preprocess(df_raw, "text")
    // transform clean tokens to features
    val result = Vectorizers.CountVect(df_clean, "clean_tokens", "features")
    // spit data to train and test
    val Array(train, test) = result.randomSplit(Array(0.8, 0.2))
    val model = Models.TrainSVM(train)

    val predictions = model.transform(test)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("class")
      .setPredictionCol("predicted")
      .setMetricName("f1")
    val f1 = evaluator.evaluate(predictions)
    println("Test set f1 = " + f1)

    val evalPrecision = new MulticlassClassificationEvaluator()
      .setLabelCol("class")
      .setPredictionCol("predicted")
      .setMetricName("weightedPrecision")

    val precision = evalPrecision.evaluate(predictions)
    println(s"Precision = $precision")

    val evalRecall = new MulticlassClassificationEvaluator()
      .setLabelCol("class")
      .setPredictionCol("predicted")
      .setMetricName("weightedRecall")

    val recall = evalRecall.evaluate(predictions)
    println(s"Recall = $recall")

  }

  //
  def Eval24HoursPred() = {
    Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
    val spark = SparkSession.builder()
      .appName("Spark sess")
      .master("local[*]")
      .getOrCreate()

    val schema = StructType(
      Array(StructField("text", StringType),
        StructField("pred", DoubleType),
        StructField("hand", DoubleType),
      )
    )

    val df_raw = spark
      .read
      .option("header", "true")
      .schema(schema)
      .csv("src/24hours/out.csv")

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("hand")
      .setPredictionCol("pred")
      .setMetricName("f1")

    val f1 = evaluator.evaluate(df_raw)
    println("Test set f1 = " + f1)

  }

  object Models {
    def TrainSVM(train: Dataset[Row]): LinearSVCModel = {
      val lsvm = new LinearSVC()
        .setMaxIter(10)
        .setRegParam(0.1)
        .setLabelCol("class")
        .setPredictionCol("predicted")
        .setFeaturesCol("features")

      val vcmodel = lsvm.fit(train)
      vcmodel
    }

    def TrainRF(train: Dataset[Row]): RandomForestClassificationModel = {
      val rf = new RandomForestClassifier()
        .setLabelCol("class")
        .setPredictionCol("predicted")
        .setFeaturesCol("features")
        .setMaxDepth(5)
        .setNumTrees(100)
        .setFeatureSubsetStrategy("auto")

      val model = rf.fit(train)
      model
    }

    def TrainLR(train: Dataset[Row]): LogisticRegressionModel = {
      val lr = new LogisticRegression()
        .setMaxIter(100)
        .setRegParam(0.2)
        .setElasticNetParam(0.8)
        .setLabelCol("class")
        .setPredictionCol("predicted")
        .setFeaturesCol("features")
        .setFamily("multinomial")

      val lrModel = lr.fit(train)
      lrModel
    }

    def TrainNB(train: Dataset[Row]): NaiveBayesModel = {
      val nb = new NaiveBayes()
        .setLabelCol("class")
        .setPredictionCol("predicted")
        .setFeaturesCol("features")

      val model = nb.fit(train)
      model
    }
  }

  def TrainModel(): Unit = {
    val spark = SparkSession.builder()
      .appName("Spark sess")
      .master("local[*]")
      .getOrCreate()

    val schema = StructType(
      Array(StructField("Id", DoubleType),
        StructField("class", DoubleType),
        StructField("text", StringType),
      )
    )

    val df_raw = spark
      .read
      .option("header", "true")
      .schema(schema)
      .csv("dataset")
    val df_clean = Preprocess(df_raw, "text")

    val countVect: CountVectorizer = new CountVectorizer()
      .setInputCol("clean_tokens")
      .setOutputCol("features")
      .setVocabSize(2000)
      .setMinDF(5)

    val lsvm = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)
      .setLabelCol("class")
      .setPredictionCol("predicted")
      .setFeaturesCol("features")

    val pipeline = new Pipeline()
      .setStages(Array(countVect, lsvm))

    val model = pipeline.fit(df_clean)

    model.write.overwrite().save("mymodel")
  }

  object Vectorizers {
    def CountVect(df: Dataset[Row], clean_tokens: String, features: String): Dataset[Row] = {
      val cvModel: CountVectorizerModel = new CountVectorizer()
        .setInputCol(clean_tokens)
        .setOutputCol(features)
        .setVocabSize(2000)
        .setMinDF(5)
        .fit(df)
      cvModel.transform(df)
    }

    def Word2Vec(df: Dataset[Row], clean_tokens: String, features: String): Dataset[Row] = {
      val word2Vec = new Word2Vec()
        .setInputCol(clean_tokens)
        .setOutputCol("unscaledfeatures")
        .setVectorSize(3)
        .setMinCount(0)
      val w2v = word2Vec.fit(df)
      val df2 = w2v.transform(df)

      val scaler = new MinMaxScaler()
        .setInputCol("unscaledfeatures")
        .setOutputCol(features)
      val scalerModel = scaler.fit(df2)
      scalerModel.transform(df2)
    }

    def HashingTF(df: Dataset[Row], clean_tokens: String, features: String): Dataset[Row] = {
      val hashingTF = new HashingTF()
        .setInputCol(clean_tokens)
        .setOutputCol("raw_features")
        .setNumFeatures(2000)

      val featurizedData = hashingTF.transform(df)
      val idf = new IDF().setInputCol("raw_features").setOutputCol(features)
      val idfModel = idf.fit(featurizedData)
      idfModel.transform(featurizedData)
    }
  }

  def Preprocess(corpus: Dataset[Row], textColumn: String): Dataset[Row] = {
    var a = RemoveFuncs.Links(corpus, textColumn)
    a = RemoveFuncs.Aliases(a, textColumn)
    a = RemoveFuncs.Hashtags(a, textColumn)
    a = RemoveFuncs.Punctuation(a, textColumn)
    a = RemoveFuncs.Spaces(a, textColumn)
    a = RemoveFuncs.MakeLowercase(a, textColumn)
    a = RemoveFuncs.ShortWords(a, textColumn)
    a = RemoveFuncs.Tokenize(a, textColumn)
    a = RemoveFuncs.StopWords(a, "tokens", "clean_tokens")
    a = RemoveFuncs.ShortWords(a, textColumn)
    a.drop("tokens").drop("text")

  }

  object RemoveFuncs {
    def Punctuation(corpus: Dataset[Row], textColumn: String): Dataset[Row] = {
      corpus.withColumn(textColumn, regexp_replace(corpus(textColumn), "[^a-zA-Z ]", replacement = " "))
    }

    def ShortWords(corpus: Dataset[Row], textColumn: String): Dataset[Row] = {
      corpus.withColumn(textColumn, regexp_replace(corpus(textColumn), "\\b\\w{1,2}\\b\\s?", replacement = ""))

    }

    def Links(corpus: Dataset[Row], textColumn: String): Dataset[Row] = {
      corpus.withColumn(textColumn, regexp_replace(corpus(textColumn), "^(http[s]?://www\\.|http[s]?://|www\\.)", ""))
    }

    def Spaces(corpus: Dataset[Row], textColumn: String): Dataset[Row] = {
      corpus.withColumn(textColumn, regexp_replace(trim(corpus(textColumn)), "\"|\\s{2,}", " "))
    }

    def Hashtags(corpus: Dataset[Row], textColumn: String): Dataset[Row] = {
      corpus.withColumn(textColumn, regexp_replace(trim(corpus(textColumn)), "#([^\\s]*)", ""))
    }

    def Aliases(corpus: Dataset[Row], textColumn: String): Dataset[Row] = {
      corpus.withColumn(textColumn, regexp_replace(trim(corpus(textColumn)), "[@]+.+?\\b", ""))
    }

    def MakeLowercase(corpus: Dataset[Row], textColumn: String): Dataset[Row] = {
      corpus.withColumn(textColumn, lower(corpus(textColumn)))
    }

    def StopWords(corpus: Dataset[Row], inputColumn: String, outputColumn: String): Dataset[Row] = {
      //      val worlds = Source.fromResource("stopWords.txt").getLines.flatMap(_.split("\\W+")).toArray
      val stopWordsRemover = new StopWordsRemover()
        //        .setStopWords(worlds)
        .setInputCol(inputColumn)
        .setOutputCol(outputColumn)
      stopWordsRemover.transform(corpus)
    }

    def Tokenize(corpus: Dataset[Row], textColumn: String): Dataset[Row] = {
      val tok = new Tokenizer()
      tok
        .setInputCol(textColumn)
        .setOutputCol("tokens")
        .transform(corpus)
    }
  }

}
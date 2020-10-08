import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession

import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.sql.functions._

object Globals {
  val PRODUCTION = true // Training mode: PRODUCTION = false, Streaming mode: PRODUCTION = true
  val DATASET = "datasets/StanfordSentimentTreebank" // Path to the downloaded dataset for training
  val TRAIN_FRACTION = 0.8 // Fraction of data getting to the train part of a split. All other is for testing.
  val SEED = 12345
}

object Main {
  def remove_path(path: String, spark: SparkSession): Unit = {
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val outPutPath = new Path(path)
    if (fs.exists(outPutPath))
      fs.delete(outPutPath, true)
  }

  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
    val spark = SparkSession.builder.appName("Streamer")
      .config("spark.master", "local[2]").getOrCreate

    remove_path("outputLogisticRegression/", spark)
    remove_path("outputRandomForest/", spark)
    remove_path("logisticRegressionCheckpoint/", spark)
    remove_path("randomForestCheckpoint/", spark)
    remove_path("singleOutput/", spark)

    if (Globals.PRODUCTION) {
      streaming(spark)
    } else {
      training()
    }
  }

  def training(): Unit = {
    // Read the dataset
    val data = Reader.read(Globals.DATASET)

    // Train/Test split
    var Array(train, test) = data.randomSplit(Array(Globals.TRAIN_FRACTION, 1 - Globals.TRAIN_FRACTION), seed = Globals.SEED)

    // Preprocess
    train = Preprocessing.preprocess(train)
    test = Preprocessing.preprocess(test)

    // Extract features with Word2vec
    val word2vecModel = FeatureExtractor.trainModel(train)
    train = FeatureExtractor.getVector(word2vecModel, train)
    test = FeatureExtractor.getVector(word2vecModel, test)

    // Train Logistic regression and Random forest classifiers
    val logRegModel = Classifier.trainLogisticRegression(train)
    val randForestModel = Classifier.trainRandomForest(train)

    Classifier.evaluateModel(logRegModel, test)
    Classifier.evaluateModel(randForestModel, test)
  }

  def streaming(spark: SparkSession): Unit = {
    // Connect to data source
    val line = spark.readStream
      .format("socket")
      .option("host", "10.90.138.32")
      .option("port", 8989)
      .load().toDF("SentimentText")


    // Load pre-trained models
    val word2vecModel = FeatureExtractor.loadModel()
    val logRegModel = Classifier.loadLogRegModel()
    val randForestModel = Classifier.loadRandomForestModel()

    // Extract features out of preprocessed tweet from the stream
    val filtered_words = word2vecModel.transform(Preprocessing.filterData(line))

    // Predict the sentiment of the tweet
    val predictions1 = logRegModel.transform(filtered_words)
    val predictions2 = randForestModel.transform(filtered_words)

    // Print the predictions
    val query1 = predictions1
      .withColumn("timestamp", current_timestamp())
      .select("timestamp", "SentimentText", "prediction")
      .writeStream
      .format("csv")
      .option("checkpointLocation", "logisticRegressionCheckpoint/")
      .option("path", "outputLogisticRegression/")
      .start()

    val query2 = predictions2
      .withColumn("timestamp", current_timestamp())
      .select("timestamp", "SentimentText", "prediction")
      .writeStream
      .format("csv")
      .option("checkpointLocation", "randomForestCheckpoint/")
      .option("path", "outputRandomForest/")
      .start()

    outputs_join_thread(spark)
    query1.awaitTermination()
    query2.awaitTermination()
  }


    res.start()
    res.join()
  }
}

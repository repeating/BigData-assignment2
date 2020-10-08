import java.nio.file.Paths

import Preprocessing._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{concat_ws, _}

import scala.reflect.io.File

object Tweets {

  def main(args: Array[String]): Unit = {
    //remove info loggers
    Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    //if project runs not for the first time
    val projectPath = Paths.get("").toAbsolutePath
    val output = Paths.get(projectPath + "/src/output/").toString
    if (File(output).exists)
      File(output).deleteRecursively()

    //create session
    val spark: SparkSession = SparkSession.builder()
      .master("local[*]")
      .appName("tweets")
      .getOrCreate()
    //get trained model/ train it
    val model = PipelineModel.load("mymodel")

    //get tweets
    spark.sparkContext.setLogLevel("ERROR")
    var df = spark
      .readStream
      .format("socket")
      .option("host", "10.90.138.32")
      .option("port", "8989")
      .load()

    // get "clear" words from twir
    val processedDF = Preprocess(df, "value")
      .select("clean_tokens").filter("size(clean_tokens)!=0")
    // get dataframe from predicted words
    val predicted = model.transform(processedDF)

    // streaming dataframes don't have common column, so create it
    val time = current_timestamp()
    val dfTime = df.withColumn("timestampDF", time)
    val predictedTime = predicted.withColumn("timestampPR", time)

    // and join them
    val joined = dfTime.join(predictedTime, dfTime("timestampDF") === predictedTime("timestampPR"))
      .select(
        col("timestampDF"),
        col("value"),
        col("predicted"))

    //saving tweets, time, decision as csv file
    val res = joined
      .writeStream
      .format("csv")
      .option("header", "false")
      .option("format", "append")
      .option("checkpointLocation", "src/output/checkpoints/")
      .option("path", "src/output/result")
      .outputMode("append")
      .start()

    // transform array of words to separate words with their count
    val transferredProcessedDF = processedDF
      .withColumn("tmp", concat_ws(",", processedDF("clean_tokens")))
      .drop("clean_tokens")
      .withColumnRenamed("tmp", "clean_tokens")

    val clearWords = transferredProcessedDF
      .select(
        explode(split(transferredProcessedDF("clean_tokens"), ",")).alias("word"))
    // get top 10 most used words
    val wordCounts = clearWords
      .groupBy(col("word"))
      .count()
      .orderBy(desc("count"))
      .limit(10)
    // save current list in memory
    val wordQuery = wordCounts
      .writeStream
      .queryName("wordCount")
      .outputMode("complete")
      .format("memory")
      .start()

    //save results to next folder, update list on timer
    val timer = new java.util.Timer()
    val folder = Paths.get(projectPath + "/src/output/wordCount").toString

    val task = new java.util.TimerTask {
      def run() = {
        //delete if already exists
        if (File(folder).exists)
          File(folder).deleteRecursively()
        //save
        spark.sql(s"select * from wordCount").coalesce(1)
          .write
          .format("com.databricks.spark.csv")
          .csv("src/output/wordCount")
      }
    }
    // write each 1.1 minutes - average time of getting twit
    timer.schedule(task, 0, 70 * 1000)
    //do it untill it will get tire you
    spark.streams.awaitAnyTermination()

  }

}
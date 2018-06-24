from collections import namedtuple
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
import sys

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, HashingTF, IDF, Word2Vec
reload(sys)
sys.setdefaultencoding('utf-8')
# from pyspark.sql.functions import desc


sc = SparkContext("local[2]", "Tweet Streaming App")



ssc = StreamingContext(sc, 10)
sqlContext = SQLContext(sc)

ssc.checkpoint( "file:/home/ubuntu/tweets/checkpoint/")

socket_stream = ssc.socketTextStream("10.0.1.4", 5555) # Internal ip of  the tweepy streamer


lines = socket_stream.window(20)
lines.pprint()
fields = ("text")
Tweet = namedtuple( 'Tweet', fields )

def getSparkSessionInstance(sparkConf):
    if ("sparkSessionSingletonInstance" not in globals()):
        globals()["sparkSessionSingletonInstance"] = SparkSession \
            .builder \
            .config(conf=sparkConf) \
            .getOrCreate()
    return globals()["sparkSessionSingletonInstance"]

def do_something(time, rdd):
    print("========= %s =========" % str(time))
    try:

        # Get the singleton instance of SparkSession
        spark = getSparkSessionInstance(rdd.context.getConf())
        model = PipelineModel.load("/home/ubuntu/server/logregCount.model")
        # Convert RDD[String] to RDD[Tweet] to DataFrame
        rowRdd = rdd.map(lambda w: Tweet(w))
        linesDataFrame = spark.createDataFrame(rowRdd)
        linesDataFrame.show()
        # Creates a temporary view using the DataFrame
        linesDataFrame.createOrReplaceTempView("tweets")

        # Do tweet character count on table using SQL and print it
        lineCountsDataFrame = spark.sql("select text as text from tweets")
        lineCountsDataFrame.show()
        p=model.transform(linesDataFrame)

        lineCountsDataFrame.show()
        print(type(p))
        p.show()

        p.select(["text","prediction"]).coalesce(1).write.format("com.databricks.spark.csv").save("dirwithcsv")

    except Exception as e:
        print (e)



# key part!
lines.foreachRDD(do_something)

ssc.start()

ssc.awaitTermination(100)

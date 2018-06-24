
import sys
import os
import findspark
findspark.init()
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import col
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, HashingTF, IDF, Word2Vec
from pyspark.ml.classification import LogisticRegression,LogisticRegressionModel

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

def method(traindata,testdata,model):
    regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
    add_stopwords = ["http", "https", "amp", "rt", "t", "c", "the"]

    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(
        add_stopwords)

    label_stringIdx = StringIndexer(inputCol="airline_sentiment", outputCol="label")
    if(model=="Count"):
        x = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)

    elif(model=="Word2Vec"):
        x=Word2Vec(vectorSize=1000, minCount=5, inputCol="filtered", outputCol="features")

    else:
        hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
        x = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms

    lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)


    if (model=="TFIDF"):
        pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, label_stringIdx, hashingTF,x,lr])
    else:
         pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, label_stringIdx,x,lr])

    pipelineFit = pipeline.fit(traindata)
    predictions=pipelineFit.transform(testdata)

    predictions.filter(predictions['prediction'] == 0).select("text", "airline_sentiment", "probability", "label",
                                                              "prediction").orderBy("probability",
                                                                                    ascending=False).show(
        n=10, truncate=30)
    predictions.filter(predictions['prediction'] == 1).select("text", "airline_sentiment", "probability", "label",
                                                              "prediction").orderBy("probability",
                                                                                    ascending=False).show( n=10, truncate=30)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label")
    print("F1: %g" % (evaluator.evaluate(predictions)))
    c="logreg"+model+".model"
    pipelineFit.save(c)



sc = SparkContext()
sqlContext = SQLContext(sc)
data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('Tweets.csv')
#print(type(data))
#exit()
drop_list = ['airline_sentiment', 'text']
data = data.select([column for column in data.columns if column in drop_list])
data.show(5)

print (data.count())

l = ["positive", "negative", "neutral"]
data = data.where(data.airline_sentiment.isin(l))
data.printSchema()
data = data.na.drop(thresh=2)

(trainingData, testData) = data.randomSplit([0.7, 0.3], seed = 100)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))
data.groupBy("airline_sentiment").count().orderBy(col("count").desc()).show()
data.groupBy("text").count().orderBy(col("count").desc()).show()


method(trainingData,testData,"Count")

method(trainingData,testData,"Word2Vec")

method(trainingData,testData,"TFIDF")

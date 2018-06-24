# DataWarehouseAssignment3

In this assignment we first created temp.py file which trains three models from Tweets.csv data and saves the model. Here for all the three models we have used Logistic Regression but for first model we have used CountVectorizer, for second one we used Word2Vec and for third one we have used TFIDF. We then created the pipeline and fir the csv data to it. We then save all the models in respective folders.

We have then checked the accuracy for each of the models on our test data created from Tweets.csv data. Based on this we have selected the model with highest accuracy i.e. Logistic Regression with CountVectorizer. In order to train our models please run file using the below commands. Please check we have used Python 2.7 in our assignments.

```
python temp.py
```
After training our models and saving it in respective folder we can load it on any file. Then we created tweeter_stream.py file in order to get tweets from twitter. We have used private host of our azure machine with port 5555. You can run this file using the below commands.

```
python tweepy_stream.py &
```
After running this file we are able to connect to twitter and fetch tweets. Now we want to get all tweets into spark stream. We have created file spark_stream_listner.py file which takes the twitter stream and converts it into dataframe. So for each batch of rdd it converts it into SQL Dataframe and feeds it into our model created in above steps. We load our model and transform the tweets and save it in csv file. Please check we get our results in tweetcsvDir directory. In order to run this file please type below commands.

```
$SPARK_HOME/bin/spark-submit spark_stream_listner.py
```
Please make sure you have spark spark-2.3.0-bin-hadoop2.7 in the same folder with values of all variables set properly. Moreover, we have used python 2.7 in our assignment. Please check the values of our variables is as given:

***export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/***

***export SPARK_HOME=~/server/spark-2.3.0-bin-hadoop2.7/***

***export PYSPARK_PYTHON=python***

We have also included Tweets.csv file so please include the file in same directory where you have kept all the python files in order to run the program. Please change private ip address in tweepy_stream.py and spark_stream_listner.py file before running the code.

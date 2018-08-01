from sklearn import svm

import pandas as pd
from sklearn.cross_validation import train_test_split

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import re

from sklearn.feature_selection import SelectKBest,chi2
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

def clean_data(dataset):
    text_data=[]
    for data in dataset:
        punctuation = '^%$#@{}|\/~`!?,.:;"\')(_-'
        newstring = data.translate(None, punctuation)
        newstring=re.sub(r'\d+', '', newstring).strip()
        text_data.append(newstring)
    return text_data

def method(features, classifier,Train_Data,Train_label,Test_Data):

    if classifier == "RandomForest":
        clf = RandomForestClassifier()
        pipeline = Pipeline([("features", features), ("rf", clf)])

    elif classifier == "DecisionTree":
        clf = DecisionTreeClassifier()
        pipeline = Pipeline([("features", features), ("dt", clf)])

    elif classifier == "NaiveBayes":
        pipeline = MultinomialNB()

    else:
        clf = SVC(kernel="linear")
        pipeline = Pipeline([("features", features), ("svm", clf)])

    pipeline.fit(Train_Data, Train_label)
    predicted = pipeline.predict(Test_Data)

    return predicted


def method2(Test_label,predicted,classifier):

    print "Accuracy of ",classifier, " ", accuracy_score(Test_label, predicted)
    cm = confusion_matrix(Test_label, predicted)
    sn.heatmap(cm, annot=True)
    plt.show()


def method3(traindata,svd,vectorizer):
    languages = traindata["label"].drop_duplicates()

    sample_df = pd.DataFrame([])
    for i in languages:
        sample_df = sample_df.append(traindata[traindata["label"] == i].sample(300))
    traindata = vectorizer.fit_transform(sample_df["clean_text"])
    cc=svd.fit_transform(traindata)
    sn.pairplot(pd.DataFrame(cc))
    plt.show()


dataset=pd.read_csv("train.txt",delimiter="\t",header=None,names=["text","label"])
se=pd.Series(clean_data(dataset["text"]))
dataset["clean_text"]=se.values

vectorizer = CountVectorizer()
traindata = vectorizer.fit_transform(dataset["clean_text"])


# PCA (svd) on the data
svd = TruncatedSVD(n_components=30)

# method3(dataset,svd,vectorizer)

select = SelectKBest(chi2, k=15)

combination_features = FeatureUnion([("TrunSvd", svd), ("chisquare", select)])

Train_Data, Test_Data ,Train_label, Test_label = train_test_split(traindata, dataset["label"],
                                                        test_size=0.3)

predicted = method(combination_features, "DecisionTree",Train_Data,Train_label,Test_Data)
method2(Test_label,predicted,"DecisionTree")

predicted = method(combination_features, "RandomForest",Train_Data,Train_label,Test_Data)
method2(Test_label,predicted,"RandomForest")

predicted = method(combination_features, "NaiveBayes",Train_Data,Train_label,Test_Data)
method2(Test_label,predicted,"NaiveBayes")

dataForSvm = dataset.sample(n=7000)
vec = vectorizer.fit_transform(dataForSvm["clean_text"])
Train_Data, Test_Data ,Train_label, Test_label = train_test_split(vec, dataForSvm["label"],test_size=0.3)
predicted = method(combination_features, "SVM",Train_Data,Train_label,Test_Data)
method2(Test_label,predicted,"SVM")



# grid search for SVM

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
clf = svm.SVC()

grid_search = GridSearchCV(clf,parameters)
grid_search.fit(Train_Data, Train_label)
print("\n Best configuration for %s after grid search have the following parameter: \n",
          grid_search.best_estimator_)


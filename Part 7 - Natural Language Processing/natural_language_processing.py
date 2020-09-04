"""
@author: Raghav
"""
#Importing Libraries
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

#Importing Dataset
dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t',quoting=3);

#Cleaning the texts
import re;
import nltk;
nltk.download('stopwords');
from nltk.corpus import stopwords;
from nltk.stem.porter import PorterStemmer;
corpus=[];
all_stopwords=stopwords.words('english');
all_stopwords.remove('not');
for i in range(0,len(dataset)):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i]);
    review=review.lower();
    review=review.split();
    ps=PorterStemmer();
    review=[ps.stem(word) for word in review if not word in set(all_stopwords)];
    review=' '.join(review);
    corpus.append(review);

#Creating Bag of Models
from sklearn.feature_extraction.text import CountVectorizer;
cv=CountVectorizer(max_features=1500);
X=cv.fit_transform(corpus).toarray();
y=dataset.iloc[:,-1].values;

#Splitting Dataset
from sklearn.model_selection import train_test_split;
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0);

#Training Naive Based Model on Training Set
from sklearn.naive_bayes import GaussianNB;
classifier=GaussianNB();
classifier.fit(X_train,y_train);

#Predicting Test Result
y_pred=classifier.predict(X_test);
# print(np.concatenate((y_test.reshape(len(y_test),1),y_pred.reshape(len(y_pred),1)),1));

#Confusion Matrix
from sklearn.metrics import confusion_matrix;
cm=confusion_matrix(y_test,y_pred);
print(cm);
from sklearn.metrics import accuracy_score;
print(accuracy_score(y_test, y_pred));


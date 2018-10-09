import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('spam.csv',encoding='latin-1')
dataset = dataset.iloc[:,0:2].values
y=dataset[:,0]

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,5572):
    message = re.sub('[^a-zA-Z]', ' ',dataset[i][1])
    message = message.lower()
    message = message.split()
    ps = PorterStemmer()
    message = [ps.stem(word) for word in message if not word in set(stopwords.words('english'))]
    message=' '.join(message)
    corpus.append(message)

 #Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

#encoding y
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
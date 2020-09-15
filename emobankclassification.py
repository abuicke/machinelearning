import datetime
print(datetime.datetime.now())

import os
import math
import numpy
import re
import pickle
import pandas
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

emobank = pandas.read_csv("data/2000-emobank.csv")
# Should be able to use triplets or just 3-sized arrays for V.A.D.?
X, y = numpy.asanyarray(emobank['text']), numpy.asanyarray(emobank['V'])

documents = []

stemmer = WordNetLemmatizer()

for line in X:
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(line))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)
    
vectorizer = CountVectorizer()#stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(str(datetime.datetime.now()) + ' Finished generating train/test sets')
print(str(datetime.datetime.now()) + ' Fitting data to random forest regressor...')

regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

print(str(datetime.datetime.now()) + ' Running predictions...')
y_pred = regressor.predict(X_test)

print(y_pred)

meanAbsoluteError = mean_absolute_error(y_test, y_pred)
print("MAE:", meanAbsoluteError)
meanSquaredError = mean_squared_error(y_test, y_pred)
print("MSE:", meanSquaredError)
rootMeanSquaredError = math.sqrt(meanSquaredError)
print("RMSE:", rootMeanSquaredError)

with open('models/emobank-regressor', 'wb') as picklefile:
    pickle.dump(regressor, picklefile)
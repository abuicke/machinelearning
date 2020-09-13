import matplotlib.pyplot as plt
import pandas
import pylab
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pandas.read_csv("emobank.csv")

# Split data set into train and test
msk = numpy.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

regr = LinearRegression()
train_x = numpy.asanyarray(train[['V']])
train_y = numpy.asanyarray(train[['D']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

test_x = numpy.asanyarray(test[['V']])
test_y = numpy.asanyarray(test[['D']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % numpy.mean(numpy.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % numpy.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y))
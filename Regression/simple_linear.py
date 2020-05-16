import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

# read data 
df = pd.read_csv("StudentsPerformance.csv")


# mean, std, min,max of data
print(df.describe())


# take a look at the dataset
print(df.head(10))


# customizes the data set
cdf = df[['gender','parental level of education','test preparation course','math_score','reading_score','writing_score']]
print(cdf.head(9))

# plot histogram 
viz = cdf[['reading_score','math_score','writing_score']]
viz.hist()
plt.show()

# plot writing score vs math score sccater plot 
plt.scatter(cdf.math_score, cdf.writing_score,  color='blue')
plt.xlabel("math_score")
plt.ylabel("writing_score")
plt.show()


"""
Creating train and test dataset
Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. 
After which, you train with the training set and test with the testing set. This will provide a more accurate evaluation on 
out-of-sample accuracy because the testing dataset is not part of the dataset that have been used to train the data. 
It is more realistic for real world problems.

This means that we know the outcome of each data point in this dataset, making it great to test with! And since this data has 
not been used to train the model, the model has no knowledge of the outcome of these data points. So, in essence, it is truly an 
out-of-sample testing.

Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% for testing. We create a mask 
to select random rows using np.random.rand() function:
"""
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

"""
Simple Regression Model
Linear Regression fits a linear model with coefficients  𝜃=(𝜃1,...,𝜃𝑛)  to minimize the 'residual sum of squares'
between the independent x in the dataset, and the dependent y by the linear approximation.
"""

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['math_score']])
train_y = np.asanyarray(train[['writing_score']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


"""
Plot outputs
we can plot the fit line over the data:
"""

plt.scatter(train.math_score, train.writing_score,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("math_score")
plt.ylabel("writing_score")
plt.show()


"""
Evaluation
we compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics provide 
a key role in the development of a model, as it provides insight to areas that require improvement.

There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set:

Mean absolute error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand 
since it’s just average error.
Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean absolute 
error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger 
errors in comparison to smaller ones.
Root Mean Squared Error (RMSE): This is the square root of the Mean Square Error.
R-squared is not error, but is a popular metric for accuracy of your model. It represents how close the data are to the 
fitted regression line. The higher the R-squared, the better the model fits your data. Best possible score is 1.0 and it can 
be negative (because the model can be arbitrarily worse).
"""
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['math_score']])
test_y = np.asanyarray(test[['writing_score']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )




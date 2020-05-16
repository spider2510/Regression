import matplotlib.pyplot as pyplot
import pandas as pd
import pylab as pl
import numpy as np

# read data 
df = pd.read_csv("StudentsPerformance.csv")

# take a look at the dataset
print(df.head())

# customizes the data set
cdf = df[['gender','parental level of education','test preparation course','math_score','reading_score','writing_score']]
print(cdf.head(9))

# plot writing score vs math score sccater plot 
plt.scatter(cdf.math_score, cdf.writing_score,  color='blue')
plt.xlabel("math_score")
plt.ylabel("writing_score")
plt.show()


"""
Creating train and test dataset
Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. 
After which, you train with the training set and test with the testing set. This will provide a more accurate evaluation 
on out-of-sample accuracy because the testing dataset is not part of the dataset that have been used to train the data. 
It is more realistic for real world problems.

This means that we know the outcome of each data point in this dataset, making it great to test with! And since this 
data has not been used to train the model, the model has no knowledge of the outcome of these data points. So, in essence, 
it’s truly an out-of-sample testing.
"""


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# Train data distribution
plt.scatter(cdf.math_score, cdf.writing_score,  color='blue')
plt.xlabel("math_score")
plt.ylabel("writing_score")
plt.show()


"""
Multiple Regression Model
In reality, there are multiple variables that predict the Co2emission. When more than one independent variable is present, 
the process is called multiple linear regression. For example, predicting co2emission using FUELCONSUMPTION_COMB, EngineSize and 
Cylinders of cars. The good thing here is that Multiple linear regression is the extension of simple linear regression model.
"""


from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['reading_score','math_score']])
y = np.asanyarray(train[['writing_score']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)


"""
As mentioned before, Coefficient and Intercept , are the parameters of the fit line. Given that it is a multiple linear 
regression, with 3 parameters, and knowing that the parameters are the intercept and coefficients of hyperplane, sklearn can 
estimate them from our data. Scikit-learn uses plain Ordinary Least Squares method to solve this problem.
"""



"""Ordinary Least Squares (OLS)¶
OLS is a method for estimating the unknown parameters in a linear regression model. OLS chooses the parameters of a linear 
function of a set of explanatory variables by minimizing the sum of the squares of the differences between the target dependent 
variable and those predicted by the linear function. In other words, it tries to minimizes the sum of squared errors (SSE) or 
mean squared error (MSE) between the target variable (y) and our predicted output ( 𝑦̂  ) over all samples in the dataset.

OLS can find the best parameters using of the following methods:

- Solving the model parameters analytically using closed-form equations
- Using an optimization algorithm (Gradient Descent, Stochastic Gradient Descent, Newton’s Method, etc.)
"""


y_hat= regr.predict(test[['reading_score','math_score']])
x = np.asanyarray(test[['reading_score','math_score']])
y = np.asanyarray(test[['writing_score']])
print("Residual sum of squares: %.2f"% np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


"""
explained variance regression score:
If  𝑦̂   is the estimated target output, y the corresponding (correct) target output, and Var is Variance, the square of the 
standard deviation, then the explained variance is estimated as follow:

𝚎𝚡𝚙𝚕𝚊𝚒𝚗𝚎𝚍𝚅𝚊𝚛𝚒𝚊𝚗𝚌𝚎(𝑦,𝑦̂ )=1−𝑉𝑎𝑟{𝑦−𝑦̂ }𝑉𝑎𝑟{𝑦} 
The best possible score is 1.0, lower values are worse.
"""

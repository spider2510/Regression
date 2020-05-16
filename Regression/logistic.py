"""
What is the difference between Linear and Logistic Regression?¶
While Linear Regression is suited for estimating continuous values (e.g. estimating house price), it is not the best tool for 
predicting the class of an observed data point. In order to estimate the class of a data point, we need some sort of guidance on 
what would be the most probable class for that data point. For this, we use Logistic Regression.
"""


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt


churn_df = pd.read_csv('ChurnData.csv')
print(churn_df.head(8))

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
print(churn_df.head())


churn_df.shape
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
print(X[0:5])

y = np.asarray(churn_df['churn'])
print(y[0:5])


#  normalize the dataset:

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])


"""
Train/Test dataset
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

"""
Modeling (Logistic Regression with Scikit-learn)
Lets build our model using LogisticRegression from Scikit-learn package. This function implements logistic regression and can 
use different numerical optimizers to find parameters, including ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers. You 
can find extensive information about the pros and cons of these optimizers if you search it in internet.

The version of Logistic Regression in Scikit-learn, support regularization. Regularization is a technique used to solve the 
overfitting problem in machine learning models. C__ parameter indicates __inverse of regularization strength which must be a 
positive float. Smaller values specify stronger regularization. Now lets fit our model with train set:
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
print(LR)

yhat = LR.predict(X_test)
print(yhat)

"""
predict_proba returns estimates for all classes, ordered by the label of classes. So, the first column is the probability of 
class 1, P(Y=1|X), and second column is probability of class 0, P(Y=0|X):
"""

yhat_prob = LR.predict_proba(X_test)
# print(yhat_prob)


"""
Evaluation¶
jaccard index
Lets try jaccard index for accuracy evaluation. we can define jaccard as the size of the intersection divided by the size of the 
union of two label sets. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the 
subset accuracy is 1.0; otherwise it is 0.0.
"""


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)

"""
confusion matrix
Another way of looking at accuracy of classifier is to look at confusion matrix.
"""

from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print(confusion_matrix(y_test, yhat, labels=[1,0]))



# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix

plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
plt.show()
print (classification_report(y_test, yhat))


""""
log loss
Now, lets try log loss for evaluation. In logistic regression, the output can be the probability of customer churn is yes 
(or equals to 1). This probability is a value between 0 and 1. Log loss( Logarithmic loss) measures the performance of a 
classifier where the predicted output is a probability value between 0 and 1.
"""

from sklearn.metrics import log_loss
print(log_loss(y_test, yhat_prob))




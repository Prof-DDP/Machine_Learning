# Logistic Regression on Iris dataset

# Importing necessary libraries
import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf) #so it shows the whole array when printing

# Importing dataset
dataset = pd.read_csv(r"iris_data.csv")

# Features and labels
X = dataset.iloc[:, :-1].values #takes all rows and columns 3 and 4. Model is only based on age and salary

y = dataset.iloc[:, -1].values #all rows and only the last column

# Encoding categorical (non-numeric) data
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

np.savetxt("X_test_iris.csv", y_test, delimiter=",")
np.savetxt("y_test_iris.csv", y_test, delimiter=",")

# Fitting classifier to dataset
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
clf = OneVsRestClassifier(LogisticRegression(random_state=0))
clf.fit(X_train, y_train)

# Evaluating
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(clf, X_train, y_train, cv=3)
#f1 = f1_score(y_train, y_pred, average='macro')
f1 = f1_score(y_train, y_pred, average='weighted') #72.92%

# Testing model w/ confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train.flatten(), y_pred.flatten()) #266 correct, 34 incorrect

# F1 and cm indicate that another model would most likely be better suited for the task

















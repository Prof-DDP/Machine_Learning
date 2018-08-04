# Simple Linear Regression

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold=np.inf) #so it shows the whole array when printing

# Importing dataset
dataset = pd.read_csv(r"Salary_Data.csv")

# Features and labels
X = dataset.iloc[:, :-1].values #takes all rows and all except the last column

y = dataset.iloc[:, -1].values #all rows and only the last column

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Feature scaling
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #transform both but only fit to X_train so that X_train and X_test have same scale
#scaling not needed for y_train and test b/c the dependant variable (y) doesn't have a wide range of possiblities in this problem like it could have in like regression
'''

# Fitting model to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting Test set results
y_pred = regressor.predict(X_test)

# Data visualization (training)
plt.scatter(X_train, y_train, color='red')
plt.xlabel('Experience (yrs)')
plt.ylabel('Salary ($)')
plt.title('Salary vs. Experience (Training set)')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()

# Data visualization (testing)
plt.scatter(X_test, y_test, color='red')
plt.xlabel('Experience (yrs)')
plt.ylabel('Salary ($)')
plt.title('Salary vs. Experience (Test set)')
plt.plot(X_train, regressor.predict(X_train), color='blue') #didn't change this line b/c this is comparing the trained regression line against the test set
plt.show()















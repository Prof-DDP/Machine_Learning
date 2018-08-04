# Multiple Linear regression on breast cancer dataset

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold=np.inf) #so it shows the whole array when printing

# Importing dataset and changing missing data
dataset = pd.read_csv(r"breast-cancer-wisconsin.data.csv")
dataset = dataset.replace('?', np.NaN)

# Features and labels
X = dataset.iloc[:, :-1].values #takes all rows and all except the last column

y = dataset.iloc[:, -1].values #all rows and only the last column

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0) 
X = imputer.fit_transform(X)

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple linear regression model to data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Backward Elimination time!
import statsmodels.formula.api as sm

X = np.append(np.ones((699, 1)).astype(float), X, 1)

X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 4, 5, 6, 7, 8, 9]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 4, 5, 7, 8, 9]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 4, 7, 8, 9]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

# New training data w/ optimized X matrix
X_train_opt, X_test_opt , y_train, y_test = train_test_split(X_opt, y, test_size=0.2, random_state=0)

# Regression w/ optimized X
regressor_opt = LinearRegression()
regressor_opt.fit(X_train_opt, y_train)

y_pred_opt = regressor_opt.predict(X_test_opt)

# Evaluating acc of predictions

from model_scoring import Multiple_Linear_Regression as MLR
MLR.y_eval(y_pred, y_pred_opt, y_test)














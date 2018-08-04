#Multiple Linear Regression

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold=np.inf) #so it shows the whole array when printing

# Importing dataset
dataset = pd.read_csv(r"50_Startups.csv")

# Features and labels
X = dataset.iloc[:, :-1].values #takes all rows and all except the last column

y = dataset.iloc[:, -1].values #all rows and only the last column

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding dummy variable trap
X = X[:, 1:] #Libraries do this for you but why not do it now

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting
y_pred = regressor.predict(X_test)

# Building optimal model using Backward Elimination
import statsmodels.formula.api as sm

#adding column of ones to account for x0 in multiple linear regression equation
#appending X to column of ones and not other way around because ones column must be first

X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]] #includes everything at first. Show this way because certain ones will be removed later

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() #OLS stands for Ordinary Least Squares (estimates params in model)
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]] #removed feature w/ highest P-value after summary
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() 
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]] #removed feature w/ highest P-value after summary
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() 
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]] #removed feature w/ highest P-value after summary
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() 
regressor_OLS.summary()

X_opt = X[:, [0, 3]] #removed feature w/ highest P-value after summary
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() 
regressor_OLS.summary()

# Using optimized X
#overall results are better but not great by any means
X_train_opt, X_test_opt , y_train, y_test = train_test_split(X_opt, y, test_size=0.2, random_state=0)

regressor_opt = LinearRegression()
regressor_opt.fit(X_train_opt, y_train)

y_pred_opt = regressor_opt.predict(X_test_opt)

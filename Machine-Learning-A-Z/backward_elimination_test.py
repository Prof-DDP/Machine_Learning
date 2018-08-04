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
X_opt = X[:, ]

'''
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    num_vars = len(x[0])
    for i in range(0, num_vars):
        regressor_OLS = sm.OLS(y, x).fit()
        max_var = max(regressor_OLS.pvalues).astype(float)
        if max_var > sl:
            for j in range(0, num_vars - i):
                if (regressor_OLS.pvalues[j].astype(float) == max_var):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
'''


def backwardElimination(x, sl):
    x_og = x
    previous_x = x
    last_rsquared = 0
    num_vars = len(x[0])
    deleted_vars = []
    for i in range(num_vars):
        regressor_OLS = sm.OLS(y,x).fit()
        max_var = max(regressor_OLS.pvalues).astype(float)
        rsquared_val = regressor_OLS.rsquared_adj.astype(float)
        regressor_OLS.summary()
        
        if i == 0 and last_rsquared <= rsquared_val or last_rsquared <= rsquared_val:
            print(last_rsquared, rsquared_val)
            if max_var > sl:
                for j in range(0, num_vars - i):
                    if (regressor_OLS.pvalues[j].astype(float) == max_var):
                        previous_x = x
                        x = np.delete(x, j, 1)
                        deleted_vars.append(x_og[:, j])
                        last_rsquared = rsquared_val
        else:
            deleted_vars = np.array(deleted_vars)
            return previous_x, deleted_vars
            
    
SL = 0.05
new_X, deleted = backwardElimination(X_opt, SL)


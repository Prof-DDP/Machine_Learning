# Support Vector Regression

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold=np.inf) #so it shows the whole array when printing

# Importing dataset
dataset = pd.read_csv(r"Position_Salaries.csv")

# Features and labels
X = dataset.iloc[:, 1:2].values #Make sure X is seen as matrix and not vector. Index 0 not included because it has no more relevant information than position level

y = dataset.iloc[:, -1].values #all rows and only the last column
y = y.reshape(-1,1) #Make y a column instead of a row

#No need to split into training and test set w/ such small amounts of observations

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting SVR to dataset 
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') #rbf = gaussian
regressor.fit(X, y)


#predicting new result w/ Polynomial regression
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]])))) #Transforming desired prediction into array and then to scale. Then inversing the transformation

#visualizing SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#visualizing SVR results (higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

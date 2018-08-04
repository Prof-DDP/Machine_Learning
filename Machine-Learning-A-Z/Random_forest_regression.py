# Random Forest Regression (non-linear, non-continuous, ensemble)

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

# Fitting Regressor to dataset 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X,y)

#predicting new result w/ Polynomial regression
y_pred = regressor.predict(6.5) #w/ 10 trees = 167,000; w/ 100 trees = 158,300; w/ 300 trees = 160,333!

#visualizing Regression results (higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) #interval of 0.01 for best representation of data
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
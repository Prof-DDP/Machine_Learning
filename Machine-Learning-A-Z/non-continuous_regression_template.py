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

# Splitting the dataset into Training set and Test set
'''
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
'''

# Feature scaling
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #transform both but only fit to X_train so that X_train and X_test have same scale
#scaling not needed for y_train and test b/c the dependant variable (y) doesn't have a wide range of possiblities in this problem like it could have in like regression
'''

# Fitting Regressor to dataset 

#predicting new result w/ Polynomial regression
y_pred = regressor.predict(6.5)

# Visualizing Regression results (higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) #interval of 0.01 for best representation of data
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
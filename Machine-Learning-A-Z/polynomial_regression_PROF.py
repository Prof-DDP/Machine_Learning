# Polynomial Regression

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

#No need to split into training and test set w/ such small amounts of observations

# Fitting Linear Regression to dataset (for comparison)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) #play with degree value to see different results
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Comparing regressors

#visualizing linear regression results

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
 
#visualizing Polynomial regression results

X_grid = np.arange(min(X), max(X), 0.1) #so resulting plot line has more contiuous curve
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#predicting new result w/ Linear regression
lin_reg.predict(6.5) #predicts salary of employee w/ position level 6.5. result ~ 330,378.787878787879

#predicting new result w/ Polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5)) #result ~ 158,862.45265153

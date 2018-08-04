# Polynomial Regression from scratch (Gradient Descent vs. Normal Equation)

# Importing core libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-whitegrid')

np.random.seed(42)

# Features and labels
sample = 100
X = 5 * np.random.rand(sample, 1) - 2.5
y = 4 * (X**2) + X + 3 + np.random.randn(sample,1)

# Hyperparameters
initial_m = 0
initial_m2 = 0
initial_b = 0
theta = [initial_b, initial_m, initial_m2]

eta=0.05
iterations=1000

# Finding theta that minimizes cost function
X_poly = np.c_[X, X**2]
X_poly = np.c_[np.ones((sample,1)), X_poly]
theta_best = np.linalg.inv( X_poly.T.dot(X_poly) ).dot(X_poly.T).dot(y)

b = theta_best[0]
m = theta_best[1]
m2 = theta_best[2]

y_pred = m2*X**2 + m*X + b

# Training w/ Gradient Descent
def step_gradient(X,y,m2,m,b,learning_rate):
    m2_gradient = 0
    m_gradient = 0
    b_gradient = 0
    for i in range(sample):
        m2_gradient += -2*X[i]**2 * ( y[i] - m2*X[i]**2 + m*X[i] + b )
        m_gradient += -2*X[i] * ( y[i] - m2*X[i]**2 + m*X[i] + b )
        b_gradient += -2 * ( y[i] - m2*X[i]**2 + m*X[i] + b )
    
    m2 -= ( m2_gradient / sample ) * learning_rate
    m -= ( m_gradient / sample ) * learning_rate
    b -= ( b_gradient / sample ) * learning_rate
    
    return m2, m, b

def gradient_descent(X,y,m2,m,b):
    for i in range(iterations):
        m2_gd, m_gd, b_gd = step_gradient(X,y,m2,m,b,eta)
    return m2_gd, m_gd, b_gd

m2_gd, m_gd, b_gd = gradient_descent(X,y,initial_m2,initial_m,initial_b)        

y_pred_gd = m2_gd*X**2 + m_gd*X + b_gd

# Evaluating 
def compute_error(X,y,m2,m,b):
    se = 0
    for i in range(sample):
        se += (y[i] - (m2 * X[i]**2 + m * X[i] + b))**2
    rmse = np.sqrt( se / sample )
    return rmse

initial_error = compute_error(X,y,initial_m2,initial_m,initial_b)

final_error = compute_error(X,y,m2,m,b)
final_error_gd = compute_error(X,y,m2_gd,m_gd,b_gd)

# Visualizing results
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(X,y,c='orange')
ax.scatter(X, y_pred, c='b')
ax.scatter(X, y_pred_gd, c='r')

ax.set_title('Toy polynomial data')
ax.set_xlabel('X')
ax.set_ylabel('y')
plt.show()

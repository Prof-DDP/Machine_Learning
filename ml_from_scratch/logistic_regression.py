# Logistic Regression from scratch

# Importing core libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-whitegrid')

np.random.seed(42)

# Features and labels
class0_xvals = (8 + np.random.rand(50)).reshape(-1,1)
class1_xvals = (9 + np.random.rand(50)).reshape(-1,1)

class0_yvals = (8.9 + np.random.randn(50)).reshape(-1,1)
class1_yvals = (3.92 + np.random.randn(50)).reshape(-1,1)

X = np.c_[ np.vstack((class0_xvals, class1_xvals)), np.vstack((class0_yvals, class1_yvals)) ]
X = np.c_[ np.ones(X.shape[0]), X ]

y = np.array([0.0 if val in class0_xvals else 1.0 for val in X])

# Hyperparameters
initial_theta = np.zeros(X.shape[1])
eta = 0.1
iterations = 2000

# Training model w/ Gradient Descent
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def compute_accuracy(X,y,theta):
    h = np.round( sigmoid( np.dot(X, theta) ) )
    
    return (( h == y ).sum().astype(float) / y.size)

def step_gradient(X,y,theta,lr):
    y_hat = np.dot(X, theta)
    h = sigmoid(y_hat)
    theta_gradient = np.dot(X.T , (h - y)) / y.size
    
    theta -= (theta_gradient * lr)
    
    return theta

def gradient_descent(X,y,theta,lr):
    initial_acc = compute_accuracy(X,y,theta)
    
    for i in range(iterations):
        theta = step_gradient(X,y,theta,lr)
    
    final_acc = compute_accuracy(X,y,theta)
    
    return theta,initial_acc,final_acc

theta,initial_acc,final_acc = gradient_descent(X,y,initial_theta,eta)

initial_theta = np.zeros(X.shape[1])

def predict(X, theta):
    threshold = 0.5
    probs = sigmoid(np.dot(X, theta))
    predictions = np.array([1 if prob >= threshold else 0 for prob in probs])
    
    return probs, predictions

initial_y_probs, initial_y_pred = predict(X, initial_theta)

y_probs, y_pred = predict(X, theta)

# Visualizing results
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(211)
'''
for i in range(len(X)):
    if y[i] == 0:
        ax.scatter(X[i, 1], X[i, 2], c='orange')
    else:
        ax.scatter(X[i, 1], X[i, 2], c='b')
'''
ax.scatter(X[initial_y_pred == y][:, 1], X[initial_y_pred == y][:, 2], c='g', alpha=0.8, label='Correct')
ax.scatter(X[initial_y_pred != y][:, 1], X[initial_y_pred != y][:, 2], c='r', alpha=0.8, label='Incorrect')
plt.legend(loc='upper right')
ax.set_title('Toy linearly separable data (Initial theta predictions)', fontsize=12)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('y', fontsize=12)

ax = fig.add_subplot(212)
ax.scatter(X[y_pred == y][:, 1], X[y_pred == y][:, 2], c='g', alpha=0.8, label='Correct')
ax.scatter(X[y_pred != y][:, 1], X[y_pred != y][:, 2], c='r', alpha=0.8, label='Incorrect')
plt.legend(loc='upper right')
ax.set_title('Toy linearly separable data (Final theta predictions)', fontsize=12)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('y',fontsize=12)
plt.show()


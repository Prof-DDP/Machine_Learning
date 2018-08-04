# Logistic Regression from scratch. Building model as Object instead of separate functions

# Importing core libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-whitegrid')

np.random.seed(42)

# Features and Labels
class0_vals = np.hstack( (20 + np.random.randn(100,1), 22 + np.random.rand(100,1)) )
class1_vals = np.hstack( (20.5 + np.random.randn(100,1), 20.63 + np.random.rand(100,1)) )

X = np.vstack( (class0_vals, class1_vals) )
X = np.c_[ np.ones(X.shape[0]), X ]

y = np.array([0.0 if X[i, 1:] in class0_vals else 1.0 for i in range(X.shape[0])])

'''
class0_xvals = (8 + np.random.rand(50)).reshape(-1,1)
class1_xvals = (9 + np.random.rand(50)).reshape(-1,1)

class0_yvals = (8.9 + np.random.randn(50)).reshape(-1,1)
class1_yvals = (3.92 + np.random.randn(50)).reshape(-1,1)

X = np.c_[ np.vstack((class0_xvals, class1_xvals)), np.vstack((class0_yvals, class1_yvals)) ]
X = np.c_[ np.ones(X.shape[0]), X ]

y = np.array([0.0 if val in class0_xvals else 1.0 for val in X])
'''

# Hyperparameters
initial_theta = np.zeros(X.shape[1])
eta = 0.01
iterations = 60000

# Defining Model
class LogisticRegression:
    def __init__(self,theta,lr,iters):
        self.initial_theta = np.zeros(X.shape[1])
        self.theta = theta
        self.lr = lr
        self.iters = iters
    
    def add_b0(self,x):
        x = np.c_[ np.ones(x.shape[0]), x ]
        
        return x
    
    def check_x(self,x):
        if not set(X[:, 0] == np.ones(X.shape[0])) == {True}:
            x = self.add_b0(x)
            
        return x
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def compute_accuracy(self,X,y,theta):
        h = np.round( self.sigmoid( (np.dot(X, theta)) ) )
        
        return (( h == y ).sum().astype(float) / y.size)
    
    def step_gradient(self,X,theta,lr):
        z = np.dot( X, theta )
        h = self.sigmoid(z)
        theta_gradient = np.dot( X.T, (h - y)) / y.size
        
        theta -= (theta_gradient * lr)
        
        return theta
    
    def gradient_descent(self):
        initial_acc = self.compute_accuracy(self.X, self.y, self.initial_theta)
        
        for i in range(self.iters):
            self.theta = self.step_gradient(self.X,self.theta,self.lr)
        
        final_acc = self.compute_accuracy(self.X, self.y, self.theta)
        
        return initial_acc,final_acc
    
    def fit(self,X,y):
        self.X = self.check_x(X)
        self.y = y
        self.initial_acc, self.final_acc = self.gradient_descent()
    
    def predict(self):
        initial_probs = self.sigmoid( np.dot(self.X, self.initial_theta) )
        initial_y_pred = np.round( initial_probs )
        
        probs = self.sigmoid( np.dot(self.X, self.theta) )
        y_pred = np.round( probs )
        
        return probs, y_pred, initial_probs, initial_y_pred
    
clf = LogisticRegression(initial_theta,eta,iterations)
clf.fit(X,y)
theta, initial_acc, final_acc = clf.theta, clf.initial_acc, clf.final_acc #Best acc = 0.93. eta=0.01, iterations=60000, theta=[ 36.57129642,   0.98556982,  -2.59573815]
probs, y_pred, initial_probs, initial_y_pred = clf.predict()

initial_theta = np.zeros(X.shape[1])

# Visualizing Results
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



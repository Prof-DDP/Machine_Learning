# Linear Regression from scratch (Using Gradient Descent)

# Importing core libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-whitegrid')

# Features and labels
X = 2 + np.random.rand(50)
y = 3*X + 7 + np.random.rand(50)

# Hyperparameters
initial_m = 0
initial_b = 0
eta = 0.1
iterations = 2000 #Found through trial and error. Was 100 initially

# Training w/ Gradient Descent
def compute_error(X,y,m,b):
    se = 0
    N = len(X)
    for i in range(N):
        se +=  (y[i] - (m*X[i] + b) ) ** 2
    mse = se / N
    
    return mse

def step_gradient(X,y,m,b,learning_rate):
    m_gradient = 0
    b_gradient = 0
    N = len(X)
    
    for i in range(N):
        m_gradient += -2*X[i] * (y[i] - (m*X[i] + b))
        b_gradient += -2 * (y[i] - (m*X[i] + b))
    
    m -= (m_gradient / N) * learning_rate
    b -= (b_gradient / N) * learning_rate
        
    return m,b

def gradient_descent(m,b,num_iterations):
    initial_error = compute_error(X,y,m,b)
    
    for i in range(num_iterations):
        m, b = step_gradient(X,y,m,b,eta)
    
    final_error = compute_error(X,y,m,b)
    
    return m,b,initial_error,final_error

m,b,initial_error,final_error = gradient_descent(initial_m, initial_b,iterations)

# Results w/ other iter values
m2,b2,initial_error2,final_error2 = gradient_descent(initial_m, initial_b,100)
m3,b3,initial_error3,final_error3 = gradient_descent(initial_m, initial_b,500)
m4,b4,initial_error4,final_error4 = gradient_descent(initial_m, initial_b,1000)

# Comparing model to sklearn equivalent
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(np.array(X).reshape(-1,1),y)

sklearn_error = compute_error(X,y,regressor.coef_,regressor.intercept_)

# Visualizing results
if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(211)
    
    ax.scatter(X,y,c='b')
    ax.plot(X, (m*X + b), c='g')
    ax.plot(X, (m2*X + b2), c='k')
    ax.plot(X, (m3*X + b3), c='m')
    ax.plot(X, (m4*X + b4), c='orange')
    ax.set_title('Toy linear data')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    
    ax = fig.add_subplot(212)
    ax.plot(X, regressor.predict(np.array(X).reshape(-1,1)), c='r')
    ax.scatter(X,y,c='b')
    ax.set_title('Toy linear data')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    
    plt.show()
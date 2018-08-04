# The more universally applicable aspects of data preprocessing

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold=np.inf) #so it shows the whole array when printing

# Importing dataset
dataset = pd.read_csv(r"Data.csv")

# Features and labels
X = dataset.iloc[:, :-1].values #takes all rows and all except the last column

y = dataset.iloc[:, -1].values #all rows and only the last column

# Encoding categorical (non-numeric) data
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) #encoding into numbers 
#France = column 0, Germany = column 1, Spain = column 2
'''

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

'''
# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #transform both but only fit to X_train so that X_train and X_test have same scale
#scaling not needed for y_train and test b/c the dependant variable (y) doesn't have a wide range of possiblities in this problem like it could have in like regression
'''



















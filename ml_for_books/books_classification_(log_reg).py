# Logistic regression on scraped book data
# Author: Prof D. (4/21/18)

# Importing necessary libraries
import numpy as np
import pandas as pd

# Importing dataset
df = pd.read_csv(r'book_data.csv')

# Features and labels
X = df.drop(['rating'], 1).values
y = df['rating'].values

# Splitting into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting clf to data
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Evaluating
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

y_scores = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv=5)
mean, std = y_scores.mean(), y_scores.std() #Mean: 21.7%, Std: 0.019

y_pred = cross_val_predict(clf,X_train,y_train,cv=5)

f1_score_val = f1_score(y_train, y_pred, average='weighted') #0.13
recall_score_val = recall_score(y_train,y_pred,average='weighted') #0.2
precision_score_val = precision_score(y_train,y_pred,average='weighted') #0.228

cm = confusion_matrix(y_train,y_pred)

#Model not performing well. Low variance but high bias. Data is most likely non-linear
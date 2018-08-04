# KNN on iris dataset

# Importing necessary libraries
import pandas as pd

# Importing dataset
dataset = pd.read_csv(r"iris_data.csv")

# Features and labels
X = dataset.iloc[:, :-1].values #takes all rows and columns 3 and 4. Model is only based on age and salary

y = dataset.iloc[:, -1].values #all rows and only the last column

#Encoding categorical variable 
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

# Splitting
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Feature scaling
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test) #transform both but only fit to X_train so that X_train and X_test have same scale
'''

# Fitting classifier to dataset
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2))
clf_scaled = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2))

clf.fit(X_train, y_train)
'''
clf_scaled.fit(X_train_scaled, y_train)
'''
# Evaluating
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(clf, X_train, y_train, cv=3)
'''

y_pred_scaled = cross_val_predict(clf, X_train_scaled, y_train, cv=3)
'''
'''
Weighted performs better

f1_macro = f1_score(y_train, y_pred, average='macro')
'''

f1_weighted = f1_score(y_train, y_pred, average='weighted') #95.999%


#Scaling impacts results negatively
'''
f1_macro_scaled = f1_score(y_train, y_pred_scaled, average='macro')
f1_weighted_scaled = f1_score(y_train, y_pred_scaled, average='weighted')
'''

# Testing model w/ confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train.flatten(), y_pred.flatten()) #292 correct and 8 incorrect
'''
cm_scaled = confusion_matrix(y_train.flatten(), y_pred_scaled.flatten())
'''

# Model looks good. Time to apply to test set
y_pred_test = clf.predict(X_test)
cm_final = confusion_matrix(y_test.flatten(), y_pred_test.flatten()) #148 correct and 2 incorrect
f1_final = f1_score(y_test, y_pred_test, average='weighted') #98%

# Saving model
from sklearn.externals import joblib
joblib.dump(clf, 'Knn_iris.pkl')
'''
Knn_iris = joblib.load('Knn_iris.pkl')
'''















#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading the libraries
dataset = pd.read_csv('8. Model_Selection\Data.csv')
X = dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#Spliting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.25,random_state=1)

#Features Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training the model with training set
from sklearn.ensemble import RandomForestClassifier
#Decision Tree based on gini
clf_gini = RandomForestClassifier(n_estimators=10,criterion='gini',random_state=12)
clf_gini.fit(X_train,y_train)
#Decision Tree based on entropy
clf_entropy = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=12)
clf_entropy.fit(X_train,y_train)

#Predicting the value
y_pred_gini = clf_gini.predict(X_test)
y_pred_entropy = clf_entropy.predict(X_test)

#Finding the accuracy and confusion matrics
from sklearn.metrics import confusion_matrix, accuracy_score
cm_gini = confusion_matrix(y_test, y_pred_gini)
print("accuracy and confusion matrics of Random Forest Classifier based on gini:-")
print(cm_gini)
acc_gini = accuracy_score(y_test, y_pred_gini)
print(acc_gini*100)

from sklearn.metrics import confusion_matrix, accuracy_score
cm_entropy = confusion_matrix(y_test, y_pred_entropy)
print("\naccuracy and confusion matrics of Random Forest Classifier based on entropy:-")
print(cm_entropy)
acc_entropy = accuracy_score(y_test, y_pred_entropy)
print(acc_entropy*100)
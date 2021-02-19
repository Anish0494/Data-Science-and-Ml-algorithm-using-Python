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
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train,y_train)

#Predicting the value
y_pred = clf.predict(X_test)

#Finding the accuracy and confusion matrics
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc = accuracy_score(y_test, y_pred)
print(acc*100)
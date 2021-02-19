#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Multiple Linear Regression\Startups.csv')
print(dataset.head())
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

#Encoding Independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x=np.array(ct.fit_transform(x))
print(x)

#Splitting the dataset into the training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 0)

#training the multiple linear regression model on the training set
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)

#Predicting the test set results
y_pred = lr.predict(X_test)
y_pred = np.array(y_pred)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.values.reshape(len(y_test),1)), axis=1))

#Evaluating the model
from sklearn.metrics import r2_score
print("/n")
print(r2_score(y_test,y_pred)*100)
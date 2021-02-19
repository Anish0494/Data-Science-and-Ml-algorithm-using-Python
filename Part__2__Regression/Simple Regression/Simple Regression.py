#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the data
dataset=pd.read_csv('Simple Regression\Salary_Data.csv')
print(dataset.head())
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#Visualising the dataset
plt.scatter(X,Y)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#Spliting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=1)

#Training the Simple Linear Regression model on the training set
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
print("Coef = ",lr.coef_)
print("Intercept = ",lr.intercept_)

#Predicting the Test set results
y_pred = lr.predict(X_test)

#Visualising the training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train,lr.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the testing set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test,y_pred,color='blue')
plt.title('Salary vs Experience (Testing set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

#Evaluating the model
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred)*100)
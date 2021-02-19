#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Polynomial Regression\Position_Salaries.csv')
print(dataset)
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Training the Linear Regression Model on the whole dataset
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)

#Training the polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=6)
x_poly = poly_reg.fit_transform(x)
lr_poly = LinearRegression()
lr_poly.fit(x_poly,y)

#Visulaising the linear regression Results
plt.scatter(x,y,color='red')
plt.plot(x,lr.predict(x),color='blue')
plt.title('Visulaising the linear regression Results')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising the polynomial Regression Results
plt.scatter(x,y,color='red')
plt.plot(x,lr_poly.predict(x_poly),color='blue')
plt.title('Visualising the polynomial Regression Results')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()

#Visualising the polynomail Regression results (for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, lr_poly.predict(poly_reg.fit_transform(x_grid)),color='blue')
plt.title('Visualising the polynomial Regression Results (for higher resolution and smoother curve)')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()

#Prdicting a new result with linear Regression
print(lr.predict([[6.5]]))

#Predicting a new result with polynomial Regression
print(lr_poly.predict(poly_reg.fit_transform([[6.5]])))

#Evaluating the model
from sklearn.metrics import r2_score
print(r2_score(y,lr.predict(y))*100)
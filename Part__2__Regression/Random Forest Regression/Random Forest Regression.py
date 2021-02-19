#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Regression\Random Forest Regression\Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Training the libraries with whole dataset
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 10, random_state= 0)
rfr.fit(x,y)

#Predicting the result
print(rfr.predict([[6.5]]))

#Visualisisng the Decision Tree Regression results(higher resolution)
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,rfr.predict(x_grid),color='blue')
plt.title('Random Forest Regression')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()
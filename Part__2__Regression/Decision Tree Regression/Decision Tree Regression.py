#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Regression\Decision Tree Regression\Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Training the Decision Tree Regression Model on th whole dataset
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=1)
dtr.fit(x,y)

#Predicting a new result
print(dtr.predict(x))
print(dtr.predict([[6.5]]))

#Visualisisng the Decision Tree Regression results(higher resolution)
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,dtr.predict(x_grid),color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()
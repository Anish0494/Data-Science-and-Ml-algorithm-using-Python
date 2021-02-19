#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing thr dataset and some preprocessing
dataset = pd.read_csv('Regression\SVR\Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

#Training the SVR model on whole dataset
from sklearn.svm import SVR
svr_m=SVR(kernel = 'rbf')
svr_m.fit(x,y)

#Predicting a new result
y_pred = svr_m.predict(sc_x.transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred)
print(y_pred)

#Visualising the predicted result
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color='red')
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(svr_m.predict(x)),color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualising the SVR results(for Higher resolution and smooth curve)
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(x_grid, sc_y.inverse_transform(svr_m.predict(sc_x.transform(x_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
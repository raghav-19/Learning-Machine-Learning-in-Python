"""
@author: Raghav
"""
# Decision Tree Regression is more accurately fit to multiple feature
#Importing Libraries
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

#Importing Datset
dataset = pd.read_csv("Position_Salaries.csv");
X = dataset.iloc[:,1:-1].values;
y = dataset.iloc[:,-1].values;

#Training Dataset on Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor;
regressor=DecisionTreeRegressor(random_state=0);
regressor.fit(X,y);

#Prediction using Model
y_pred=regressor.predict(X);
plt.scatter(X,y,color='blue',label = "Actual");
plt.scatter(X,y_pred,color='red', label = "Predicted");
plt.legend(loc="upper left");
plt.show();

#Visualizing Decision Tree Regression in High Resolution
X_grid=np.arange(min(X),max(X),0.1);
X_grid=X_grid.reshape(len(X_grid),1);
plt.scatter(X,y,color='blue',label='Actual');
plt.plot(X_grid,regressor.predict(X_grid),color='red',label="Predicted");
plt.legend(loc='upper left');
plt.show();
"""
@author: Raghav
"""
#Importing Libraries
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

#Importing Datset
dataset = pd.read_csv("Position_Salaries.csv");
X = dataset.iloc[:,1:-1].values;
y = dataset.iloc[:,-1].values;
# print(X);
# print(y);

#Feature Scaling(Its very important for good result from support vector algo)
y = y.reshape(len(y),1);
from sklearn.preprocessing import StandardScaler;
sc_x = StandardScaler();
sc_y = StandardScaler();
X = sc_x.fit_transform(X);
y = sc_y.fit_transform(y);
# print(X);
# print(y);

#Training Model SVR on whole dataset
from sklearn.svm import SVR;
regressor = SVR(kernel='rbf');
regressor.fit(X,y);

#Prediction the SVR Result
print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))));

#Visualizing the SVR Result
plt.scatter(sc_x.inverse_transform(X),sc_y.inverse_transform(y),color='blue',label="Actual");
plt.plot(sc_x.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X)),color='red',label="Predicted");
plt.legend(loc="upper left");
plt.show();


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

#Linear Model Prediction on Whole Dataset
from sklearn.linear_model import LinearRegression;
lin_regressor = LinearRegression();
lin_regressor.fit(X, y);
y_pred = lin_regressor.predict(X);
# plt.scatter(X,y,color='blue');
# plt.scatter(X,y_pred,color='red');
# plt.show();

#Ploynomial Regression Prediction on whole Dataset
from sklearn.preprocessing import PolynomialFeatures;
poly_mat = PolynomialFeatures(degree = 6);
X_poly = poly_mat.fit_transform(X);
poly_regressor = LinearRegression();
poly_regressor.fit(X_poly, y);
y_pred = poly_regressor.predict(X_poly);
plt.scatter(X,y,color='blue',label = "Actual");
plt.scatter(X,y_pred,color='red', label = "Predicted");
plt.legend(loc="upper left");
plt.show();

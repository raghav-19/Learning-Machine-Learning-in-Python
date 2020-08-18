"""
@author: Raghav
"""
#Importing Libraries
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

#Importing Dataset
dataset = pd.read_csv("50_Startups.csv");
X = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,-1].values;

#Encoding Categorical Data
from sklearn.compose import ColumnTransformer;
from sklearn.preprocessing import OneHotEncoder;
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough');
X = np.array(ct.fit_transform(X));

#Splitting Dataset
from sklearn.model_selection import train_test_split;
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0);

#Training Multiple Linear Regression
from sklearn.linear_model import LinearRegression;
regressor = LinearRegression();
regressor.fit(X_train,y_train);

#Predicting Test Result
y_pred = regressor.predict(X_test);
ind = np.arange(len(y_test));
plt.scatter(ind,y_test,color='blue',label="Actual");
plt.scatter(ind,y_pred,color='red',label="Predicted");
plt.legend(loc="upper left");
plt.show();

np.set_printoptions(precision=2);
final_result = np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),axis=1);
print(final_result);

#Making Single Prediction
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]));
print(regressor.coef_);
print(regressor.intercept_);

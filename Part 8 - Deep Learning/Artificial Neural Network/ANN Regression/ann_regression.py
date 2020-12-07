# -*- coding: utf-8 -*-
"""ann_regression.ipynb
#Dataset link - https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant

#Artificial Neural Network Regression Problem

##Importing Libraries and Dataset
"""

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import tensorflow as tf;

df=pd.read_excel("/content/drive/MyDrive/Dataset/Folds5x2_pp.xlsx",sheet_name='Sheet1')
X=df.iloc[:,:-1].values;
y=df.iloc[:,-1].values;

"""##Splitting Dataset and Feature Scaling"""

from sklearn.model_selection import train_test_split;
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=5)

from sklearn.preprocessing import StandardScaler;
sc=StandardScaler();
X_train=sc.fit_transform(X_train);
X_test=sc.transform(X_test);

"""##Building ANN"""

ann=tf.keras.models.Sequential();
ann.add(tf.keras.layers.Dense(6,activation='relu'));
ann.add(tf.keras.layers.Dense(6,activation='relu'));
ann.add(tf.keras.layers.Dense(1));

ann.compile(optimizer='adam',loss='mae');
history=ann.fit(X_train,y_train,batch_size=32,epochs=30,verbose=1,validation_data=(X_test,y_test));

"""##Predicting for Test dataset"""

y_pred=ann.predict(X_test);
from sklearn.metrics import mean_absolute_error;
mean_absolute_error(y_pred,y_test)

"""##Visualize Model"""

plt.plot(history.history['loss'],label='train');
plt.plot(history.history['val_loss'],label='test');
plt.ylabel("MAE");
plt.legend();
plt.plot();

ann.summary()

from tensorflow.keras.utils import plot_model;
plot_model(ann,show_shapes=True)
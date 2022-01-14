"""
@author: Raghav
"""
#Importing Libraries
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

#Importing Dataset
dataset=pd.read_csv("Social_Network_Ads.csv");
X=dataset.iloc[:,:-1].values;
y=dataset.iloc[:,-1].values;

#Splitting Dataset
from sklearn.model_selection import train_test_split;
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0);

#Feature Scaling
from sklearn.preprocessing import StandardScaler;
sc=StandardScaler();
X_train=sc.fit_transform(X_train);
X_test=sc.transform(X_test);

#Training Radial Basis Function Model
from sklearn.svm import SVC;
classifier=SVC(kernel='rbf',random_state=0);
classifier.fit(X_train,y_train);

#Checking for Test Set
y_pred=classifier.predict(X_test);
# print(np.concatenate((y_test.reshape(len(y_test),1),y_pred.reshape(len(y_pred),1)),1));

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix;
cm=confusion_matrix(y_test,y_pred);
print(cm);
from sklearn.metrics import accuracy_score;
print(accuracy_score(y_test,y_pred));

#Apply k-fold Cross Validation
from sklearn.model_selection import cross_val_score;
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10);
print("Accuracy: {:.2f}%".format(accuracies.mean()*100));
print("Standard Deviation: {:.2f}%".format(accuracies.std()*100));

#Applying Grid Search to find the best Models and Parameter
from sklearn.model_selection import GridSearchCV;
parameters=[{'C':[0.25,0.5,0.75,1],'kernel':['linear']},
            {'C':[0.25,0.5,0.75,1],'kernel':['rbf'],'gamma':[0.2,0.4,0.6,0.8]}];
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10);
grid_search.fit(X_train,y_train);
best_accuracy=grid_search.best_score_;
best_param=grid_search.best_params_;
print('Best Accuracy: {:.2f}%'.format(best_accuracy*100));
print('Best Parameters:',best_param);

dur_age=1;
dur_salary=10;
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = dur_age),
                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = dur_salary))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = dur_age),
                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = dur_salary))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Confusion Matrix Visualization
from sklearn.metrics import confusion_matrix
def heatmap_confusion_matrix(y_true,y_pred):
    cm=confusion_matrix(y_true,y_pred)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize=(6,4))
    sns.set(font_scale=1.2)
    sns.heatmap(cm,annot=labels,fmt='',cmap='Blues')
    plt.xlabel("Predicted Value")
    plt.ylabel("Actual Value")
    plt.show()


#Classification Metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
def classification_metrics(y_true,y_predict,y_probability):
    print("Accuracy Score  : ",round(accuracy_score(y_true,y_predict),3))
    print("Precision Score : ",round(precision_score(y_true,y_predict),3))
    print("Recall Score    : ",round(recall_score(y_true,y_predict),3))
    print("F1 Score        : ",round(f1_score(y_true,y_predict),3))
    print("roc_auc_score   : ",round(roc_auc_score(y_true,y_probability),3))


#Creating Plot of Feature Importance
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def plot_feature_importance(feature_val,col,model_name):
    feature_imp = pd.DataFrame(sorted(zip(feature_val,col)), columns=['Value','Feature'])
    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title(model_name+' Feature importance')
    plt.tight_layout()
    plt.show()


#Evaluate Binary Classification Model
def evaluate_model(model,train_X,test_X,train_y,test_y,name):
    print(name)
    model.fit(train_X,train_y)
    m_pred=model.predict(test_X)
    m_prob=model.predict_proba(test_X)[:,1]
    heatmap_confusion_matrix(test_y,m_pred)
    classification_metrics(test_y,m_pred,m_prob)


#Plot Histogram of each column in dataFrame
def plot_hist_each_col(dataFrame):
    col=np.floor(np.sqrt(dataFrame.shape[1]))
    row=int(dataFrame.shape[1]/col)+1
    plt.figure(figsize=(5*col,5*row));
    plt.suptitle("Histogram of Values",fontsize=20);
    for i in range(0,dataFrame.shape[1]):
        plt.subplot(row,col,i+1);
        f=plt.gca();
        f.set_title(dataFrame.columns[i]);
        vals=dataFrame.iloc[:,i].nunique();
        plt.hist(dataFrame.iloc[:,i],bins=vals);
    plt.tight_layout(rect=[0,0.03,1,0.95]);
    #plt.savefig('Histogram.jpg')


#Plot Pie Chart of each column in dataFrame
def plot_pie_each_col(dataFrame):
    col=np.floor(np.sqrt(dataFrame.shape[1]))
    row=int(dataFrame.shape[1]/col)+1
    plt.figure(figsize=(5*col,5*row));
    plt.suptitle("Pie Chart of Values",fontsize=20);
    for i in range(0,dataFrame.shape[1]):
        plt.subplot(row,col,i+1);
        f=plt.gca();
        f.set_title(dataFrame.columns[i]);
        values=dataFrame.iloc[:,i].value_counts(normalize=True).values;
        indexs=dataFrame.iloc[:,i].value_counts(normalize=True).index;
        plt.pie(values, labels=indexs);
    plt.tight_layout(rect=[0,0.03,1,0.95]);
    #plt.savefig('Pie_Chart.jpg')

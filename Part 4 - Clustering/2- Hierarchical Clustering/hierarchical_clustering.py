"""
@author: Raghav
"""
#Importing Libraries
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

#Importing Dataset
dataset=pd.read_csv("Mall_Customers.csv");
X=dataset.iloc[:,[3,4]].values;

#Using Dendrogram to find Optimal number of Cluster
import scipy.cluster.hierarchy as sch;
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'));
plt.title("Dendrogram");
plt.xlabel("Customers");
plt.ylabel("Euclidean Distance");
plt.show();
node=5;

#Training Hierarchical Clustering Model
from sklearn.cluster import AgglomerativeClustering;
hc=AgglomerativeClustering(n_clusters=node,affinity='euclidean',linkage='ward');
y_hc=hc.fit_predict(X);

#Visualize the Clusters
color=['red','green','blue','cyan','magenta'];
for c in range(0,node):
    plt.scatter(X[y_hc==c,0],X[y_hc==c,1],color=color[c],label='Cluster '+str(c+1));
plt.xlabel("Annual Income");
plt.ylabel("Spending Score");
plt.legend();
plt.show();

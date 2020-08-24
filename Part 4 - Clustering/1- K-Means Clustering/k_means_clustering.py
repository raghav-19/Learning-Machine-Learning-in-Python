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

#Using elbow method to find number of cluster
from sklearn.cluster import KMeans;
wcss=[];
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42);
    kmeans.fit(X);
    wcss.append(kmeans.inertia_);
    
plt.plot(range(1,11),wcss);
plt.title("The Elbow Method");
plt.xlabel("Number of Cluster");
plt.ylabel("WCSS");
plt.show();

#Training the K-Means Model on dataset
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42);
y_kmeans=kmeans.fit_predict(X);

#Visualize the Clusters
color=['red','green','blue','cyan','magenta'];
for c in range(0,5):
    plt.scatter(X[y_kmeans==c,0],X[y_kmeans==c,1],color=color[c],label='Cluster '+str(c+1));
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,color='black',label='Centroids');
plt.xlabel("Annual Income");
plt.ylabel("Spending Score");
plt.legend();
plt.show();








"""
@author: Raghav
"""
#Importing Libraries
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import random as rd;

#Importing Dataset
dataset=pd.read_csv("Ads_CTR_Optimisation.csv");

#Implementing Thompson Sampling
N=10000;
d=10;
selection=[0]*d;
reward=[0]*d;
ads_selected=[];
for i in range(0,N):
    ad=0;
    max_random=0;
    for j in range(0,d):
        cur_random=rd.betavariate(reward[j]+1,selection[j]-reward[j]+1);
        if cur_random>max_random:
            max_random=cur_random;
            ad=j;
    ads_selected.append(ad);
    selection[ad]+=1;
    reward[ad]+=dataset.values[i,ad];

#Visualizing Algorithm
plt.hist(ads_selected);
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

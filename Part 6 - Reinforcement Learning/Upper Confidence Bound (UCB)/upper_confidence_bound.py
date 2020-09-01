"""
@author: Raghav
"""
#Importing libraries
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import math

#Importing dataset
dataset=pd.read_csv("Ads_CTR_Optimisation.csv");

#Implementing USB
N=10000;
d=10;
ads_selected=[];
selection=[0]*d;
sum_reward=[0]*d;
for i in range(0,N):
    ad=0;
    max_upper_bound=0;
    for j in range(0,d):
        if selection[j]>0:
            average_reward=(sum_reward[j]/selection[j]);
            delta=math.sqrt(1.5*math.log2(i+1)/selection[j]);
            upper_bound=average_reward+delta;
        else :
            upper_bound=(1e100);
        if upper_bound>max_upper_bound:
            max_upper_bound=upper_bound;
            ad=j;
        
    ads_selected.append(ad);
    selection[ad]+=1;
    sum_reward[ad]+=(dataset.values[i,ad]);
    

#Visualizing Algorithm
plt.hist(ads_selected);
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

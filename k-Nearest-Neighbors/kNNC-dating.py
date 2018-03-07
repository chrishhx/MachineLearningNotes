#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:20:16 2018

@author: chris
"""

import os
os.chdir('/home/chris/workspace')

# read data
import pandas as pd
import numpy as np
data = np.array(pd.read_csv("datingTestSet.csv",header=None))
X = data[:,0:3]
X_colname = ['Frequent flier miles per year','Percentage of time playing video games','Average Consumption']
Y = data[:,3]

# scatter plot between attributes
'''
import matplotlib.pyplot as plt
legends = ['Did not like','Liked in small doses','Liked in large doses']
fig = plt.figure(figsize=[15,15])
for attr1 in range(0,3):
    for attr2 in range(0,3):
        ax = fig.add_subplot(330+attr1*3+attr2+1)
        for label in [1,2,3]:
            ax.scatter(X[np.where(Y==label),attr1],X[np.where(Y==label),attr2],alpha = 0.6)
        ax.set_xlabel(X_colname[attr1])
        ax.set_ylabel(X_colname[attr2])
        ax.legend(legends)
'''

# Z-score Standarize
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier as kNNC
from sklearn.model_selection import cross_val_score

# Normalize X
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Split train and test datasets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_scaled,Y,test_size=0.2,random_state=0)

# Choose k
scores = []
for k in range(1,20):
    knnc = kNNC(n_neighbors = k,weights='uniform')
    scores.append(np.average(cross_val_score(knnc,X_train,Y_train,cv=5,n_jobs=-1)))

best_k = range(1,20)[np.argmax(scores)]
model = kNNC(n_neighbors = best_k,weights='uniform')
model.fit(X_train,Y_train)

'''
fig = plt.figure(figsize=[5,5])
ax = fig.add_subplot(111)
ax.plot(range(1,20),scores)
ax.legend(['score'])
'''

train_acc = np.sum( model.predict(X_train) == Y_train ) / Y_train.shape[0]
test_acc = np.sum( model.predict(X_test) == Y_test ) / Y_test.shape[0]

print ('train accuracy is %.2f, test accuracy is %.2f'%(train_acc,test_acc))

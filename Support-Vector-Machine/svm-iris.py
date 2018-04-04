#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X = X[y!=2,:2]
y = y[y!=2]

svc = SVC(C=1, kernel='rbf', gamma='auto', coef0=0.0,
          shrinking=True, probability=False, tol=1e-3, cache_size=200,
          class_weight=None, verbose=True, max_iter=-1, decision_function_shape='ovr', random_state=0)

svc.fit(X,y)
svc.score(X,y)

x_min , x_max = X[:,0].min() - 0.5 , X[:,0].max() + 0.5
y_min , y_max = X[:,1].min() - 0.5 , X[:,1].max() + 0.5

xx , yy = np.mgrid[x_min:x_max:200j,y_min:y_max:200j]

Z = svc.decision_function(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(xx,yy,Z,linestyles=['--','-','--'])
ax.scatter(X[:,0],X[:,1],c=y)
ax.set_xlim(xx.min(),xx.max())
ax.set_ylim(yy.min(),yy.max())


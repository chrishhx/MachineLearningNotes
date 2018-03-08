#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.chdir('/home/chris/workspace/MachineLearningNotes/Generalized-Linear-Models')

from sklearn.datasets import load_boston

import numpy as np
data = load_boston(return_X_y=False)
print(data['DESCR'])
X = np.c_[np.ones(data['data'].shape[0]),np.matrix(data['data'])]
y = np.matrix( data['target'] ).transpose()
feature_name = data['feature_names']

# Split train and test datasets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Linear regression
coef = np.linalg.inv(X_train.transpose()*X_train) * X_train.transpose() * y_train
y_pred = X_train * coef
residues = ((y_train-y_pred).transpose() * (y_train-y_pred))

# Check whether LinearRegression has the same coef and residual
from sklearn.linear_model import LinearRegression
# X0 == 1 is add into X, so no need to fit intercept, coef[0] is the intercept
lm = LinearRegression(fit_intercept=False,normalize=False,copy_X=True)
lm.fit(X_train,y_train)
lm.coef_
lm._residues

# It's exactly the same

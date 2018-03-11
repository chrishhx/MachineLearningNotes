#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.chdir('/home/chris/workspace/MachineLearningNotes/Generalized-Linear-Models')

from sklearn.datasets import load_wine
import numpy as np

data = load_wine(return_X_y=False)
print(data['DESCR'])
X = np.c_[np.ones(data['data'].shape[0]),np.matrix(data['data'])]
y = np.matrix( data['target']-1 ).transpose()
feature_name = data['feature_names']

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
lgreg = LogisticRegression(penalty = 'l2',multi_class='ovr',solver='sag',max_iter=500)
lgreg.fit(X,y)
lgreg.score(X,y)
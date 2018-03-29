#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.chdir('/home/chris/workspace/MachineLearningNotes/Generalized-Linear-Models')
import numpy as np

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

from sklearn.linear_model import lasso_path, enet_path ,lars_path

alphas_lasso, coefs_lasso , _ = lasso_path(X_scaled,y,eps=1e-3,positive=False,fit_intercept=False)

alphas_enet, coefs_enet , _ = enet_path(X_scaled,y,eps=1e-3,positive=False,l1_ratio=0.8,fit_intercept=False)

alphas_lars, _, coefs_lars = lars_path(X_scaled,y,eps=1e-3,positive=False)

from matplotlib import pyplot as plt

fig = plt.figure(1)
fig.set_dpi(100)
ax = fig.gca()
ax.set_color_cycle (3*['b','r','m','g','y'])
l1 = plt.plot(-np.log10(alphas_lasso),coefs_lasso.T)
l2 = plt.plot(-np.log10(alphas_enet),coefs_enet.T,linestyle='--')
l3 = plt.plot(-np.log10(alphas_lars),coefs_lars.T,linestyle=':')

plt.xlabel('-Log(alpha)')  
plt.ylabel('coefficients')  
plt.title('Lasso Elastic-Net and Lars Paths')  
plt.legend((l1[-1], l2[-1], l3[-1]), ('Lasso', 'Elastic-Net' , 'Lars'), loc='lower left')
plt.axis('tight')
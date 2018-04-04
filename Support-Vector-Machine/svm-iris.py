#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris()

svc = SVC(C=1.0, kernel='rbf', gamma='auto', coef0=0.0,
          shrinking=True, probability=False, tol=1e-3, cache_size=200,
          class_weight=None, verbose=True, max_iter=-1, decision_function_shape='ovr', random_state=0)

svc.fit(iris.data,iris.target)

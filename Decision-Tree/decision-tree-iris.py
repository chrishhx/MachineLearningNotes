#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.chdir('/home/chris/workspace/MachineLearningNotes/Decision-Tree')

from sklearn.datasets import load_iris

iris = load_iris()

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='gini', splitter='best',
                             max_depth=3, min_samples_split=0.1,
                             min_samples_leaf=0.05, min_weight_fraction_leaf=0.0,
                             max_features=None, random_state=0,
                             max_leaf_nodes=None, min_impurity_decrease=0.0,
                             min_impurity_split=None, class_weight=None,
                             presort=False)
cross_val_score(dtc, iris.data, iris.target, cv=10)

import graphviz
from sklearn import tree
dtc.fit(iris.data, iris.target)
dot_data = tree.export_graphviz(dtc, out_file=None, 
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,  
                                special_characters=True)
graph = graphviz.Source(dot_data) 
graph.render("iris") 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:46:51 2020

@author: nikhildhamne
"""

from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd
from sklearn import metrics

from time import time

bc = load_breast_cancer()
print(bc)

x = scale(bc.data)
print(x)
y = bc.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
model = KMeans(n_clusters = 2, random_state = 0) # KMeans(n_clusters = 2, random_state = 0)
model.fit(x_train)
# dont pass y_train as the algorithm is going to cluster in groups without any label names

predictions = model.predict(x_test)
labels = model.labels_
print('labels', labels)
print('predictions', predictions)
print('accuracy', accuracy_score(y_test, predictions))
print('actual', y_test)

print(pd.crosstab(y_train, labels))

sample_size = 300

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    '''print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))
'''

bench_k_means(model, '1', x) 



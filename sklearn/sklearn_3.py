#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 19:10:40 2020

@author: nikhildhamne
"""

from sklearn import datasets
import numpy as np
# sklearn offers trani-test spilt:
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

# split it in features and labels
x = iris.data
y = iris.target

classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

#print(x,y)
print(x.shape)
print(y.shape)

# Train:Test => 8:2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
'''
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
'''

# creating the model
model = svm.SVC()

# train the model
model.fit(x_train, y_train)

print(model)

predictions = model.predict(x_test)
acc = accuracy_score(y_test, predictions)

print("predictions:\n", predictions)
print("actual:\n", y_test)
print("Accuracy: ", acc) 


for i in range(len(predictions)):
    print(classes[predictions[i]])
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:23:11 2020

@author: nikhildhamne
"""
from sklearn import datasets
import numpy as np
# sklearn offers trani-test spilt:
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

# split it in features and labels
x = iris.data
y = iris.target

#print(x,y)
print(x.shape)
print(y.shape)

# Train:Test => 8:2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)





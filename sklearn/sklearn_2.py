#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:44:39 2020

@author: nikhildhamne
"""

import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('car.data')
#print(data.head)

x = data[[
    'buying',
    'maint',
    'safety'
    ]].values
y = data[['class']]
#print(x,y)

# conver the data to numbers for x
le= LabelEncoder()
for i in range(len(x[0])):
    x[:,i] = le.fit_transform((x[:, i]))
    
#print(x)

# converting data for y (mapping) - dictionary
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
    }

y['class'] = y['class'].map(label_mapping)

y = np.array(y)
#print(y)

# create model

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
# train the model
knn.fit(x_train,y_train)
predictions = knn.predict(x_test)
accuracy = metrics.accuracy_score(y_test, predictions)

print("predictions: ", predictions)
print("accuracy: ", accuracy)
'''
print("actual value: ", y[20])
print("predicted value:", knn.predict(x)[20])
'''



 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 19:37:32 2020

@author: nikhildhamne
"""
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

boston = datasets.load_boston()

x = boston.data
y = boston.target

print("x: ", x)
print("x shape:", x.shape)
print("y:", y) 
print("y shape:", y.shape)

# algorithm
l_reg = linear_model.LinearRegression()

# visualise

plt.scatter(x.T[5],y)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# train

model = l_reg.fit(x_train, y_train)
predictions =  model.predict(x_test)

print("predictions:", predictions)
print("R**2:", l_reg.score(x, y))
print("coef: ", l_reg.coef_)
print("intercept:", l_reg.intercept_)






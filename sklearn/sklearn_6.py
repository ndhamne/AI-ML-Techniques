#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:37:22 2020

@author: nikhildhamne
"""
from PIL import Image
import mnist
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# create training variables
x_train = mnist.train_images()
y_train = mnist.train_lables()

x_test = mnist.test_images()
y_test = mnist.test_labels()

print("x_train", x_train)
print("x_test", x_test)
print("y_train", y_train)








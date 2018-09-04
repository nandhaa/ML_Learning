# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""

from sklearn.naive_bayes import GaussianNB
import numpy as np

#assigning predictor and target variables
x= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

#create a Gaussian Model
model = GaussianNB()

model.fit(x, Y)

predicted = model.predict([[-1, 1], [-4, 3], [1, 2]])

print (predicted)



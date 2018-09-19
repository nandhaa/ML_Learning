# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 17:53:15 2018

@author: 502491377
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


iris = datasets.load_iris()

X = iris.data[:, :2]

y = iris.target

#SVM Regularization param
C = 1.0 

svc = svm.SVC(kernel='linear', C=1,gamma=100).fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

h = (x_max / x_min)/100

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

plt.subplot(1, 1, 1)

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

plt.xlabel('Sepal length')

plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())

plt.title('SVM Linear')

plt.show()




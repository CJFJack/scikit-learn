# -*- encoding: utf-8 -*-
"""
利用scikit-learn框架的KNN算法进行分类
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2003)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
correct = np.count_nonzero((clf.predict(X_test) == y_test))
print("Accuracy is: %.3f" % (correct / len(X_test)))

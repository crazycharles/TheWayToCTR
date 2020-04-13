#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:09:11 2020

@author: charles
"""
#prepare the dataset
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
data = load_breast_cancer()
X = data.data
y = data.target
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

X_train_lst = X_train.tolist()
y_train_lst = y_train.tolist()
X_test_lst = X_test.tolist()
y_test_lst = y_test.tolist()

#
from logistic_regression import lr
import matplotlib.pyplot as plt

classifier1 = lr(max_iter = 200, alpha = 0.01)
classifier1.fit(X_train_lst,y_train_lst)
loss1, acc1 = classifier1.train_loss_acc()
loss1_x = [i+1 for i in range(len(loss1))]

classifier2 = lr(max_iter = 200, alpha = 0.001)
classifier2.fit(X_train_lst,y_train_lst)
loss2, acc2 = classifier2.train_loss_acc()
loss2_x = [i+1 for i in range(len(loss2))]

classifier3 = lr(max_iter = 200, alpha = 0.0001)
classifier3.fit(X_train_lst,y_train_lst)
loss3, acc3 = classifier3.train_loss_acc()
loss3_x = [i+1 for i in range(len(loss3))]

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.grid()
plt.plot(loss1_x, loss1, color='blue', label = 'alpha = 1e-2')
plt.plot(loss2_x, loss2, color='red', label = 'alpha = 1e-3')
plt.plot(loss3_x, loss3, color='green',label = 'alpha = 1e-4')
plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.ylim(0,0.4)
plt.legend()

plt.subplot(122)
plt.grid()
plt.plot(loss1_x, acc1, color='blue', label = 'alpha = 1e-2')
plt.plot(loss2_x, acc2, color='red', label = 'alpha = 1e-3')
plt.plot(loss3_x, acc3, color='green',label = 'alpha = 1e-4')
plt.xlabel("Number of iterations")
plt.ylabel("Accuracy")
plt.ylim(0.9,1)
plt.legend()
plt.savefig('training_loss_acc.png', dpi=300)

print("Ours: ", round(classifier1.score(X_test_lst, y_test_lst), 4))
# compare our implementation with the LR algorithm in sklearn
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=200)
print("sklearn: ", round(clf.fit(X_train, y_train).score(X_test, y_test), 4))






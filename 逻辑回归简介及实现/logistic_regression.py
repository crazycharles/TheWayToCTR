#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 08:45:19 2020

@author: charles
"""

import math
import random
class lr:
    ''' A simple implementation of logistic regression for binary problem
    Parameters
    max_iter:
    
    '''
    def __init__(self, max_iter = 1000, alpha = 0.01):
        self.max_iter = max_iter
        self.alpha = alpha
        
    def fit(self, X, y):
        '''
        The type of X and y are both list type. 
        Each row of X stands for one instance.
        Each element of y stands for one label.
        '''
        self.n_fea = len(X[0])
        self.n_ins = len(X)
        self.coef_ = [0 for _ in range(self.n_fea + 1)] # including the bias
        X = list(map(lambda x: x+[1], X)) # add 1 at the end of each instance for bias
        self._stochastic_gradient_descent(X, y)
        return self
    
    def score(self, X, y):
        '''return the accuracy of the algorithm on dataset X'''
        if len(self.coef_) > len(X[0]):
            X = list(map(lambda x: x+[1], X))
        correct_counter = 0
        for j in range(len(X)):
            wx_iter = map(lambda x, y: x * y, self.coef_, X[j])
            wx_value = sum(list(wx_iter))
            if self._sigmoid(wx_value) > 0.5:
                y_pred = 1
            else:
                y_pred = 0
            if y_pred == y[j]:
                correct_counter += 1
        return correct_counter / len(X)
    
    def predict(self, X):
        '''return the prediction list of the algorithm on dataset X'''
        if len(self.coef_) > len(X[0]):
            X = list(map(lambda x: x+[1], X))
        y_pred_lst = []
        for j in range(len(X)):
            wx_iter = map(lambda x, y: x * y, self.coef_, X[j])
            wx_value = sum(list(wx_iter))
            if self._sigmoid(wx_value) > 0.5:
                y_pred = 1
            else:
                y_pred = 0
            y_pred_lst.append(y_pred)
        return y_pred_lst 
    
    def train_loss_acc(self):
        '''record the training loss and output the graph'''
        return self.loss, self.acc
    
    def _loss(self, X, y):
        '''calculate the current loss on dataset X'''
        if len(self.coef_) > len(X[0]):
            X = list(map(lambda x: x+[1], X))
        loss = 0
        for j in range(len(X)):
            wx_iter = map(lambda x, y: x * y, self.coef_, X[j])
            wx_value = sum(list(wx_iter))
            loss += y[j] * wx_value - math.log(1+math.exp(wx_value))
        return -loss/len(X)
    
    def _stochastic_gradient_descent(self, X, y):
        '''An implementation of stochastic gradient descent
        you can replace this function with other methods i.e. gradient descent
        '''
        idx_lst = [i for i in range(self.n_ins)]
        self.loss = []
        self.acc = []
        self.loss.append(self._loss(X, y))
        self.acc.append(0)
        for i in range(self.max_iter):
            random.shuffle(idx_lst)
            for j in idx_lst:
                wx_iter = map(lambda x, y: x * y, self.coef_, X[j])
                wx_value = sum(list(wx_iter))
                error = self._sigmoid(wx_value) - y[j]
                gradient = list(map(lambda x: x * error * self.alpha, X[j]))
                self.coef_ = list(map(lambda x, y: x - y, self.coef_, gradient))
            self.loss.append(self._loss(X,y))
            self.acc.append(self.score(X,y))
            
    def _sigmoid(self,z):
        '''just return the value of sigmoid function'''
        return 1 / (1 + math.exp(-z))


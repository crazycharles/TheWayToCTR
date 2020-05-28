# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:45:19 2020

@author: an
"""

from fm import FM

from sklearn.datasets import load_boston
import tensorflow as tf
x, y = load_boston(return_X_y=True)
learning_rate = 1e-2
batch_size = 16
epoch = 1000

tf.reset_default_graph()

#initial
regressor = FM(x,y,learning_rate = learning_rate, batch_size = batch_size, epoch = epoch)
#train
regressor.fit()
#predict
#regressor.predict(x)
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:45:19 2020
python 3.6
tensorflow 1.10.0
@author: an
"""

from fm import FM

from sklearn.datasets import load_boston
import tensorflow as tf
from sklearn import preprocessing

x, y = load_boston(return_X_y=True)
x = preprocessing.scale(x)

learning_rate = 1e-3
batch_size = 16
epoch = 1000

tf.reset_default_graph()
#initial
regressor = FM(x,y,learning_rate = learning_rate, batch_size = batch_size, epoch = epoch)
#train
regressor.fit()
#predict
#regressor.predict(x)
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:32:37 2020

@author: an
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from ffm import FFM
import tensorflow as tf

# ---------------prepare data-----------------
data = pd.read_csv("train_200_converted.csv")

y = data['click']
#convert 0 1 --> -1 1
#for i in range(len(y)):
#    if y[i] == 0:
#        y[i] = -1
X = data.drop(columns="click", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.reshape([len(y_train),1])
y_test = y_test.reshape([len(y_test),1])
# ---------------model-------------------------
field_name = ['C1', 'banner_pos', 'site_category','app_domain','app_category', 'device_id', 'device_type', 'device_conn_type', 'C15', 'C16', 'C18']

tf.reset_default_graph()
clf = FFM(X_train, y_train, field_name = field_name, epoch = 500, learning_rate = 2e-3, lbd = 1e-5)
clf.fit()

y_p = clf.predict(X_test)

print("Acc on testing data: ", clf.score(X_test, y_test))
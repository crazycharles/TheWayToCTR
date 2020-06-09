# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:34:17 2020

@author: an
"""

import tensorflow as tf
import numpy as np
import os
class FFM:
    def __init__(self, x, y, field_name, lbd = 0.01, learning_rate = 2e-3, batch_size = 16, epoch = 50):
        self.x = x
        self.y = y
        self.sam_num = self.x.shape[0]
        self.fea_num = self.x.shape[1]
        self.field_name = field_name
        self.fie_num = len(self.field_name)
        self.k = self.fie_num // 2
        self.lbd = lbd
        self.learning_rate = learning_rate
        self.epoch = epoch
        
        if batch_size <= self.sam_num:
            self.batch_size = batch_size
        else:
            # modify the batch size according to the times between batch_size and sam_num
            self.batch_size = max(2, batch_size // -(-batch_size//self.sam_num))        
    
    def fit(self):
        x_input = tf.placeholder(dtype = tf.float32, shape = [None, self.fea_num], name = "x_input")
        y_input = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = "y_input")
        biases = tf.Variable(tf.random_normal(shape = [1], mean = 0, stddev = 1), name = 'biases')
        linear_w = tf.Variable(tf.random_normal(shape = [self.fea_num, 1], mean = 0, stddev = 1), name = "linear_weights")
        complex_w = tf.Variable(tf.random_normal(shape = [self.fea_num, self.fie_num, self.k], mean = 0, stddev = 1), name = "complex_weights")
    
        part1 = biases
        part2 = tf.matmul(x_input, linear_w)
        # first method to calculate the part3
        part3 = tf.Variable(tf.random_normal(shape = [1], mean = 0, stddev = 1), name = 'biases')
        for i in range(self.fea_num - 1):
            f1 = self._get_field(i)
            for j in range(i+1, self.fea_num):
                f2 = self._get_field(j)
                part3 += tf.reduce_sum(tf.multiply(complex_w[i][f2], complex_w[j][f1])) * tf.multiply(x_input[:,i],x_input[:,j])
        part3 = tf.reshape(part3,[-1, 1])
        y_predicted = tf.add(tf.add(part1, part2), part3)
        y_predicted2 = tf.nn.sigmoid(y_predicted)
        loss1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_input, logits = y_predicted))
        loss2 = tf.nn.l2_loss(complex_w)
        loss = loss1 + self.lbd * loss2
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        train_op = optimizer.minimize(loss)
        
        saver = tf.train.Saver(max_to_keep = 1)
        tf.add_to_collection('y_p', y_predicted2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                for batch_x, batch_y in self._batch():
                    sess.run(train_op, feed_dict = {x_input:batch_x, y_input:batch_y})
                if i % 50 == 0:
                    train_loss = sess.run(loss, feed_dict = {x_input: self.x, y_input: self.y})
                    print("epoch" + str(i) + "_loss: ", int(train_loss))
                    saver.save(sess, "models/m")
            sess.close()
        
    def predict(self, x_test):
        if not os.path.exists("models/"):
            print("Please train the model before test!")
            return
        
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint("models/"))

            y_ = tf.get_collection('y_p')
            y_ = sess.run(y_, feed_dict = {"x_input:0": x_test})
            y_ = np.where(y_[0]>0.5, 1, 0)
            return y_
    
    def score(self, x_test, y_test):
        if not os.path.exists("models/"):
            print("Please train the model before test!")
            return
        
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint("models/"))

            y_ = tf.get_collection('y_p')
            y_ = sess.run(y_, feed_dict = {"x_input:0": x_test})
            y_ = np.where(y_[0]>0.5, 1, 0)
            corr = 0
            for index, element in enumerate(y_):
                if element[0] == y_test[index][0]:
                    corr += 1
            return corr/len(y_test)
        
    def _batch(self):
        for i in range(0, self.sam_num, self.batch_size):
            upper_bound = min(i + self.batch_size, self.sam_num)
            batch_x = self.x[i:upper_bound]
            batch_y = self.y[i:upper_bound]
            yield batch_x, batch_y   
            
    def _get_field(self, i):
        cur = self.x.columns[i]
        for index, element in enumerate(self.field_name):
            if len(cur) > len(element) and cur[:len(element)] == element and cur[len(element)] == "_":
                return index
            
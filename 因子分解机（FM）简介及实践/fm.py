# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:18:32 2020
A simple implementation of Factorization Machines for regression problem
@author: an
"""
import tensorflow as tf
import os

class FM: 
    def __init__(self, x, y, learning_rate = 1e-6, batch_size = 16, epoch = 100):
        self.x = x
        self.y = y
        self.sam_num = x.shape[0]
        self.fea_num = x.shape[1]
        self.learning_rate = learning_rate
        self.epoch = epoch
        # the size of the vector V
        self.inner_size = self.fea_num // 2
        
        if batch_size <= self.sam_num:
            self.batch_size = batch_size
        else:
            # modify the batch size according to the times between batch_size and sam_num
            self.batch_size = max(2, batch_size // -(-batch_size//self.sam_num))
            
    def fit(self):
        x_input = tf.placeholder(tf.float32, shape=[None, self.fea_num], name = "x_input")
        y_input = tf.placeholder(tf.float32, shape=[None], name="y_input")
        biases = tf.Variable(tf.zeros([1]), name="biases")
        linear_weights = tf.Variable(tf.random_uniform(shape=[self.fea_num, 1], minval = -1, maxval = 1),name="linear_weights")
        second_order_weights = tf.Variable(tf.random_uniform(shape=[self.fea_num, self.inner_size], minval = -1, maxval = 1), name="second_order_weights")
        
        part1 = biases
        part2 = tf.reduce_mean(tf.matmul(x_input, linear_weights),0)
        part3 = 0.5 * tf.reduce_sum(tf.square(tf.matmul(x_input, second_order_weights)) -  tf.matmul(tf.square(x_input), tf.square(second_order_weights)), 1)
        y_predicted = tf.add(tf.add(part1, part2), part3)
        loss = tf.reduce_mean(tf.square(y_predicted - y_input), 0)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        train_op = optimizer.minimize(loss)
        
        saver = tf.train.Saver(max_to_keep = 1)
        tf.add_to_collection('y_p', y_predicted)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                for batch_x, batch_y in self._batch():
                    sess.run(train_op, feed_dict = {x_input:batch_x, y_input:batch_y})
                if i % 100 == 0:
                    train_loss = sess.run(loss, feed_dict = {x_input: self.x, y_input: self.y})
                    print("epoch" + str(i) + "_loss: ", int(train_loss))
                    saver.save(sess, "models/")
            sess.close()

    def predict(self, x_test):
        if not os.path.exists("models/"):
            print("Please train the model before test!")
            return
        
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('models/.meta')
            saver.restore(sess, tf.train.latest_checkpoint("models/"))
            y_ = tf.get_collection('y_p')
            print(sess.run(y_, feed_dict = {"x_input:0": x_test}))

    def _batch(self):
        for i in range(0, self.sam_num, self.batch_size):
            upper_bound = min(i + self.batch_size, self.sam_num)
            batch_x = self.x[i:upper_bound]
            batch_y = self.y[i:upper_bound]
            yield batch_x, batch_y            
    
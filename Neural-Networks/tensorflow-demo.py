# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 16:35:46 2018

@author: chris
"""

import os
os.chdir('d:/workspace')

import MNIST
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

train_labels,train_images = MNIST.read_data('train-labels.idx1-ubyte','train-images.idx3-ubyte')
test_labels,test_images = MNIST.read_data('t10k-labels.idx1-ubyte','t10k-images.idx3-ubyte')

train_labels = train_labels.astype(np.int8)
test_labels  = test_labels.astype(np.int8)

train_images = train_images.reshape(60000,784)
test_images  = test_images.reshape(10000,784)

''' check by imshow
import matplotlib.pyplot as plt
plt.imshow(test_images[0].astype(np.uint8),cmap='gray')
print(test_labels[0])
'''

input_num_units = 28*28
hidden1_num_units = 512
hidden2_num_units = 256
output_num_units = 10

x = tf.placeholder(tf.float32,[None,input_num_units])
y = tf.placeholder(tf.float32,[None,output_num_units])

epochs = 10
batch_size = 256
learning_rate = 0.01

weights = {
        'hidden1': tf.Variable(tf.random_normal([input_num_units,hidden1_num_units],seed=0)),
        'hidden2': tf.Variable(tf.random_normal([hidden1_num_units,hidden2_num_units],seed=0)),
        'output': tf.Variable(tf.random_normal([hidden2_num_units,output_num_units],seed=0))
        }

biases = {
        'hidden1': tf.Variable(tf.random_normal([hidden1_num_units],seed=0)),
        'hidden2': tf.Variable(tf.random_normal([hidden2_num_units],seed=0)),
        'output': tf.Variable(tf.random_normal([output_num_units],seed=0))
        }

hidden1_layer = tf.nn.relu(tf.add( tf.matmul(x,weights['hidden1']),
                                  biases['hidden1'] ))

hidden2_layer = tf.nn.relu(tf.add( tf.matmul(hidden1_layer,weights['hidden2']),
                                  biases['hidden2'] ))

keep_prob = tf.placeholder(tf.float32)
dropout1 = tf.nn.dropout(hidden1_layer,keep_prob)
dropout2 = tf.nn.dropout(hidden2_layer,keep_prob)

output_layer =  tf.add( tf.matmul(hidden2_layer,weights['output']),
                        biases['output'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=output_layer))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = 2 * int(train_images.shape[0] / batch_size)
        for i in range(total_batch):
            index = np.random.randint(low=0,high=train_images.shape[0],size=batch_size)
            batch_x = train_images[index]
            batch_y = train_labels[index]
            _, c = sess.run([optimizer,cost],feed_dict={x:batch_x, y:batch_y, keep_prob:0.6})
            avg_cost += c / total_batch
        
        print("\nEpoch: ",(epoch),"cost=","%.5f"%(avg_cost))
    print("\ntraining complete")
    
    pred = tf.equal(tf.argmax(output_layer,1),tf.arg_max(y,1))
    acc  = tf.reduce_mean(tf.cast(pred,'float'))
    print("train accuracy",acc.eval({x:train_images,y:train_labels}))
    print("test accuracy",acc.eval({x:test_images,y:test_labels}))
    pred = tf.argmax(output_layer,1).eval({x:test_images})

labels,images = MNIST.read_data('t10k-labels.idx1-ubyte','t10k-images.idx3-ubyte',False)
err, = np.where(pred != labels)

i = err[5]
plt.imshow(images[i].astype(np.uint8),cmap='gray')
print("true = %d , pred = %d"%(labels[i],pred[i]))

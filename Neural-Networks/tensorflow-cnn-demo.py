# -*- coding: utf-8 -*-

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

x = tf.placeholder(tf.float32,[None,28,28])
x_image = tf.reshape(x,[-1,28,28,1])
y = tf.placeholder(tf.float32,[None,10])

weights = {
        'conv1': tf.Variable(tf.truncated_normal([5,5,1,32])),
        'conv2': tf.Variable(tf.truncated_normal([5,5,32,64])),
        'hidden1': tf.Variable(tf.truncated_normal([7*7*64,1024])),
        'output' : tf.Variable(tf.truncated_normal([1024,10]))
        }

biases = {
        'conv1': tf.constant(0.1,shape=[32]),
        'conv2': tf.constant(0.1,shape=[64]),
        'hidden1': tf.constant(0,1,shape=[1024]),
        'output' : tf.constant(0,1,shape=[10])
        }

conv1 = tf.nn.relu( tf.add(tf.nn.conv2d(x_image,weights['conv1'],
                                        strides=[1,1,1,1],padding='SAME'),
                           biases['conv1']))

pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],
                       strides=[1,2,2,1],padding='SAME')

conv2 = tf.nn.relu( tf.add(tf.nn.conv2d(pool1,weights['conv2'],
                                        strides=[1,1,1,1],padding='SAME'),
                           biases['conv2']))

pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],
                       strides=[1,2,2,1],padding='SAME')

pool2_flat = tf.reshape(pool2,[-1,7*7*64])

hidden1_layer = tf.nn.relu( tf.add(tf.matmul(pool2_flat,weights['hidden1']),
                                   biases['hidden1']) )

output_layer =  tf.add( tf.matmul(hidden1_layer,weights['output']),
                        biases['output'])

keep_prob = tf.placeholder(tf.float32)
dropout1 = tf.nn.dropout(hidden1_layer,keep_prob)

epochs = 20
batch_size = 128
learning_rate = 0.01

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=output_layer))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess = tf.Session()
    sess.run(init)
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = 2 * int(train_images.shape[0] / batch_size)
        for i in range(total_batch):
            index = np.random.randint(low=0,high=train_images.shape[0],size=batch_size)
            batch_x = train_images[index]
            batch_y = train_labels[index]
            _, c = sess.run([optimizer,cost],feed_dict={x:batch_x, y:batch_y, keep_prob:0.5})
            avg_cost += c / total_batch
        
        print("\nEpoch: ",(epoch),"cost=","%.5f"%(avg_cost))
    print("\ntraining complete")
    
    pred = tf.equal(tf.argmax(output_layer,1),tf.argmax(y,1))
    acc  = tf.reduce_mean(tf.cast(pred,'float'))
    #print("train accuracy",acc.eval({x:train_images,y:train_labels},session=sess))
    print("test accuracy",acc.eval({x:test_images,y:test_labels},session=sess))
    pred = tf.argmax(output_layer,1).eval({x:test_images},session=sess)
    
labels,images = MNIST.read_data('t10k-labels.idx1-ubyte','t10k-images.idx3-ubyte',False)
err, = np.where(pred != labels)

i = err[100]
plt.imshow(images[i].astype(np.uint8),cmap='gray')
print("true = %d , pred = %d"%(labels[i],pred[i]))

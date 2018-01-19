# Tensorflow Stacked Autoencoder

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#%matplotlib inline

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNST_data/",one_hot=True)

# Paramitaers
# Graph: 784 -> 392 -> 196 -> 392 - 784
num_inputs = 784
neurons_hid1 = 392
neurons_hid2 = 196
neurons_hid3 = neurons_hid1
num_outputs = num_inputs
learning_rate = 0.01
act_func = tf.nn.relu

# Placeholder
X = tf.placeholder(tf.float32, shape=[None,num_inputs])

# Adapts the scale to the weight of the tensors
initializer = tf.variance_scaling_initializer()

# Weights
w1 = tf.Variable(initializer([num_inputs,neurons_hid1]), dtype=tf.float32)
w2 = tf.Variable(initializer([neurons_hid1,neurons_hid2]), dtype=tf.float32)
w3 = tf.Variable(initializer([neurons_hid2,neurons_hid3]), dtype=tf.float32)
w4 = tf.Variable(initializer([neurons_hid3,num_outputs]), dtype=tf.float32)

# Biasis
b1 = tf.Variable(tf.zeros(neurons_hid1))
b2 = tf.Variable(tf.zeros(neurons_hid2))
b3 = tf.Variable(tf.zeros(neurons_hid3))
b4 = tf.Variable(tf.zeros(num_outputs))

# Layers
hid_layer1 = act_func(tf.matmul(X,w1) + b1)
hid_layer2 = act_func(tf.matmul(hid_layer1,w2) + b2)
hid_layer3 = act_func(tf.matmul(hid_layer2,w3) + b3)
output_layer = act_func(tf.matmul(hid_layer3,w4) + b4)

# Loss Fuction - MSE 
loss = tf.reduce_mean(tf.square(output_layer - X))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

num_epochs = 50
batch_size = 150

# Session
with tf.Session() as sess:
    
    sess.run(init)
    
    for epoch in range(num_epochs):
        num_batches = mnist.train.num_examples // batch_size
        
        for iteration in range(num_batches):
            
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict = {X:X_batch})
            
        training_loss = loss.eval(feed_dict={X:X_batch})
        print("EPOCH: ", epoch, " LOSS: ", training_loss)
        
    saver.save(sess,'./savedModels/savedModel.ckpt')

# Test on 10 images form mnist 
num_test_images = 10

with tf.Session() as sess:

    saver.restore(sess,'./savedModels/savedModel.ckpt')
    
    results = output_layer.eval(feed_dict={X:mnist.test.images[:num_test_images]})
    
# Compare original images with their reconstructions
f, a = plt.subplots(2, 10, figsize=(20, 4))
for i in range(num_test_images):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(results[i], (28, 28)))
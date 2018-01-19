# Tensorflow Autoencoder

# Dimensionality reduction
# Feature extraction using Principal component analysis (PCA)

# Description: convert 3D cluster to 2d vis using autoencoder

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

#%matplotlib inline

# Random data in 3 dimensions
from sklearn.datasets import make_blobs
data = make_blobs(n_samples=100, n_features=3, centers= 2)

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[0])

data_x = scaled_data[:,0]
data_y = scaled_data[:,1]
data_z = scaled_data[:,2]

# Display the 3D data
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(data_x,data_y,data_z,c=data[1])

# Peramaters
num_inputs = 3
num_hidden = 2
num_outputs = num_inputs
learning_rate = 0.01

# Placeholder
X = tf.placeholder(tf.float32, shape=[None, num_inputs])

# Layers
hidden = fully_connected(X, num_hidden, activation_fn=None)
outputs = fully_connected(hidden, num_outputs, activation_fn=None)

# Loss fuction - Mean Squared Error
loss = tf.reduce_mean(tf.square(outputs - X))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train  = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# Session
num_steps = 100
with tf.Session() as sess:
    sess.run(init)
    
    for iteration in range(num_steps):
        sess.run(train,feed_dict={X: scaled_data})
    
    # Get the put put at the hidden layer b4 decoding
    output_2d = hidden.eval(feed_dict={X: scaled_data})
    
# Plot the feature extracted/compressed data
plt.scatter(output_2d[:,0], output_2d[:,1],c=data[1])
# Simple Linear Regression using Tensorflow

# Importing the libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_Y = StandardScaler()
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = sc_Y.fit_transform(y_train)
y_test = sc_Y.transform(y_test)
"""

# Reshape for tensorflow
X_train = X_train.reshape(-1)
X_test = X_test.reshape(-1)
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# Batch size
batch_size = 10

# Variables
m = tf.Variable(np.random.ranf())
b = tf.Variable(np.random.ranf())

# Placeholders
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Graph
y_model = m*X + b

# Loss Fuction - MSE 
loss = tf.reduce_mean(tf.square(Y-y_model))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# Session
with tf.Session() as sess:
    
    sess.run(init)
    
    epochs = 1000
    
    for i in range(epochs*batch_size):
        
        rand_ind = np.random.randint(len(X_train),size=batch_size)
        
        feed = {X:X_train[rand_ind], Y:y_train[rand_ind]}
        
        cost = sess.run(train,feed_dict=feed)
        
        if (i % 100) == 0:
            rand_ind = np.random.randint(len(X_train),size=batch_size)
            mse = loss.eval({X:X_train[rand_ind], Y:y_train[rand_ind]})
            print("Epoch: ", i/batch_size, "\tMSE: ", mse)
            
    model_m, model_b = sess.run([m, b])
    print("Training Done! Training cost=", cost, "m=", model_m, "b=", model_b)
    

# Training set Plot
plt.plot(X_train, y_train, 'ro', label='Training Data')
plt.plot(X_train, model_m * X_train + model_b, label='Fitted line')
plt.legend()
plt.show()

# Test set Plot
plt.plot(X_test, y_test, 'ro', label='Test Data')
plt.plot(X_train, model_m * X_train + model_b, label='Fitted line')
plt.legend()
plt.show()


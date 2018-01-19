# Tensorflow RNN

# Description: predict milk production

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

# Import the data
milk = pd.read_csv('data/monthly-milk-production.csv', index_col='Month')

#Visualiz the data
print(milk.head())
milk.index = pd.to_datetime(milk.index)
milk.plot()

# Train test split
train = milk.iloc[:-12]
test = milk.tail(n=12)

#Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train)
scaled_test = scaler.transform(test)

def next_batch(training_data,batch_size,steps):
    """
    INPUT: Data, Batch Size, Time Steps per batch
    OUTPUT: A tuple of y time series results. y[:,:-1] and y[:,1:]
    """
    
    # Set a random starting point index
    rand_start = np.random.randint(0,len(training_data)-steps) 
    
    # Index the data
    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)
    
    # Return batch
    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)


# CONSTANTS
num_inputs = 1
num_time_steps = 12
num_neurons = 100
num_outputs = 1
learning_rate = 0.03
num_train_iterations = 6000
batch_size = 1

# PLACEHOLDERS
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

# RNN CELL LAYER
cell = tf.contrib.rnn.GRUCell(num_units=num_neurons, activation=tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_outputs, reuse=tf.AUTO_REUSE)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# LOSS FUCTION -- MSE
loss = tf.reduce_mean(tf.square(outputs - y))

# OPTIMIZER
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Session
with tf.Session() as sess:
    
    sess.run(init)
    
    for iteration in range(num_train_iterations):
        
        X_batch, y_batch = next_batch(scaled_train, batch_size, num_time_steps)
        
        sess.run(train,feed_dict = {X:X_batch, y:y_batch})
        
        if iteration % 100 == 0:
            
            mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
            print(iteration,"\tMSE: ", mse)
    
    # Save Model for Later
    saver.save(sess, "./trainedModels/savedModel")

#Predicting the future
#View the future
plt.plot(test)

with tf.Session() as sess:
    
    # Restor saved model
    saver.restore(sess, "./trainedModels/savedModel")

    # Seed with last 12 months of training data
    train_seed = list(scaled_train[-12:])
    
    for iteration in range(12):
        X_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        train_seed.append(y_pred[0, -1, 0])

# portion of the results that are the generated values
results = scaler.inverse_transform(np.array(train_seed[12:]).reshape(12,1))

test['Generated'] = results
test.plot()






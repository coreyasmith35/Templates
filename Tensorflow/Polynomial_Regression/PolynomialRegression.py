# Polynomial Regression

# Importing the libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../DataSets/Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_scaled= sc_X.fit_transform(x).reshape(-1)
sc_y = StandardScaler()
y_scaled= sc_y.fit_transform(y).reshape(-1)



# Placeholders
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Graph and Variables
degree = 4
y_model = tf.Variable(tf.random_normal([1]), name='bias')
for pow_i in range(1, degree+1):
    W = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
    y_model = tf.add(tf.multiply(tf.pow(X, pow_i), W), y_model)

# Loss Fuction - MSE 
loss = tf.reduce_mean(tf.square(Y-y_model))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate= 0.001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Session
with tf.Session() as sess:
    
    sess.run(init)
    
    prev_mse = 0.0
    feed = {X:X_scaled, Y:y_scaled}
    
    epochs = 20000
    for i in range(epochs):
        
        sess.run(train,feed_dict=feed)
        
        if (i % 1000) == 0:
            mse = loss.eval({X:X_scaled, Y:y_scaled})
            print("Epoch: ", i, "\tMSE: ", mse)
        
            # Stop training when we reach a threshold
            if np.abs(prev_mse - mse) < 0.000001:
                break
        
            prev_mse = mse
     
    # Training done save a fitted line with size of x and step of 0.1
    X_grid = np.arange(min(X_scaled), max(X_scaled), 0.01)
    feed = {X:X_grid}
    y_pred = y_model.eval(feed)
    saver.save(sess,'./savedModels/savedModel.ckpt')
    print("Training Done! Training mse=", mse)
    
# Inverse transform the data
y_pred = sc_y.inverse_transform(y_pred)
X_grid = sc_X.inverse_transform(X_grid)

# Plot
plt.plot(x, y, 'ro', label='Data')
plt.plot(X_grid, y_pred, label='Fitted line')
plt.title('Polynomial Regression w/ degree %d' %degree)
plt.legend()
plt.show()


# Single pridiction on fitted line
single = 6.5

single = sc_X.transform(single).reshape(-1)
with tf.Session() as sess:
    
    saver.restore(sess,'./savedModels/savedModel.ckpt')
    
    single_pred = y_model.eval({X:single})
    
print('Single prediction at 6.5: ', sc_y.inverse_transform(single_pred)[0])
    
    
    
    
    
    
    
    
    
    

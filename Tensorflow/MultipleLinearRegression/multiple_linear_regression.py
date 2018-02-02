# Multiple Linear Regression Ternsorflow

# Importing the libraries
import tensorflow as tf
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Reshape for tensorflow
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
sc_Y = StandardScaler()
y_train = sc_Y.fit_transform(y_train)
# y_test = sc_Y.transform(y_test)

# Batch size
batch_size = 10
        
# Number of features
num_features = 5

# Variables
m = tf.Variable(tf.zeros([num_features,1]), name="b")
b = tf.Variable(tf.zeros([1]), name="b")

# Placeholders
X = tf.placeholder(tf.float32, shape=[None,num_features], name="x")
Y = tf.placeholder(tf.float32, shape=[None,1])

# Graph
y_model = tf.matmul(X, m) + b

# Loss Fuction - MSE 
loss = tf.reduce_mean(tf.square(Y-y_model))


# Optimizer
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# Session
with tf.Session() as sess:
    
    sess.run(init)
    
    epochs = 1000
    
    for i in range(epochs*batch_size):
    
        # Grab random batch of data
        # Note: does not guarantee itteration throughout whole data set
        rand_ind = np.random.randint(len(X_train),size=batch_size)
        
        feed = {X:X_train[rand_ind], Y:y_train[rand_ind]}
        
        cost = sess.run(train,feed_dict=feed)
        
        if (i % 100) == 0:
            rand_ind = np.random.randint(len(X_train),size=batch_size)
            mse = loss.eval({X:X_train[rand_ind], Y:y_train[rand_ind]})
            print("Epoch: ", i/batch_size, "\tMSE: ", mse)
     
    # Save the Graoh Values
    model_m, model_b = sess.run([m, b])
    print("Training Done! Training cost=", cost, "m=", model_m, "b=", model_b)
    
    # Predict the test set
    feed = {X:X_test}
    y_pred = y_model.eval(feed)
    
# Revert the scaling so that we can compare to y_test
y_pred = sc_Y.inverse_transform(y_pred)   
    
    
    
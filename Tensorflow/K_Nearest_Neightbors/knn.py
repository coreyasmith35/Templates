# Logistic Regression

# Importing the libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../DataSets/Social_Network_Ads.csv')
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1,1) 

# Number of input features
num_features = 2

# Number of nearest neighbors to look for
K = 5

# Placeholders
Xtr = tf.placeholder(tf.float32, [None, num_features])
Xte = tf.placeholder(tf.float32, [None])
Y = tf.placeholder(tf.float32, [None,1]) # one outpu


# Distance to all other inputs
# distance = tf.negative(tf.reduce_sum(tf.abs(tf.subtract(Xtr, Xte)),axis=1)) # L1
distance = tf.negative(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(Xtr, Xte)), reduction_indices=1))) # Euclidean

# Get the X-values and indices of the K closest neighbors
values, indices = tf.nn.top_k(distance, k=K, sorted=False)

# Retreve the values of the K closes neighbors
nearest_neighbors = tf.gather(tf.cast(Y, tf.int32), indices)

# Get most accered label
counts = tf.bincount(nearest_neighbors)
pred = tf.argmax(counts)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
accuracy = 0.
y_preds = []

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for i in range(len(X_test)):
        
        feed = {Xtr:X_train, Xte:X_test[i, :], Y:y_train}
        
        # Get the prediction for the given x value
        prediction = sess.run(pred,feed_dict=feed)
        if prediction == y_test[i]:
           accuracy += 1. / len(X_test)
           
        y_preds.append(prediction)
            
        if (i % (len(X_test)/10)) == 0:
            print('Calculating... Index:',i)
            
    print('DONE!', K, "-th neighbors' Accuracy is:", accuracy)
    

# Array of predicted values      
y_preds = np.asarray(y_preds)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
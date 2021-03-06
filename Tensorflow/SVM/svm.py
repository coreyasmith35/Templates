# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../DataSets/Social_Network_Ads.csv')
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

y[y==0] = -1

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

# Helper fuction to get next batch of data
def next_batch(X_data, y_data, batch_size):
    
     rand_ind = np.random.randint(len(X_data),size = batch_size)
     X_batch = X_data[rand_ind]
     y_batch = y_data[rand_ind]
     return X_batch, y_batch

num_features = 2
batch_size = 300
svmC = 1 # Large C = smaller-margin hyperplane

# Placeholders
X = tf.placeholder(tf.float32, [None, num_features])
Y = tf.placeholder(tf.float32, [None,1])

# Set model weights
W = tf.Variable(tf.zeros([ num_features, 1]))
b = tf.Variable(tf.zeros([1]))

# Construct model
model = tf.matmul(X, W) + b

# Loss
regularization_loss = 0.5 * tf.reduce_sum(tf.square(W)) 
hinge = tf.reduce_sum(tf.maximum(tf.zeros([batch_size,1]), 1 - Y * model))
#hinge = tf.losses.hinge_loss(Y,model)
loss = regularization_loss + svmC * hinge

# Optimization
learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# Evaluation
pred = tf.sign(model);
correct_prediction = tf.equal(Y, pred)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Initialize variables
init = tf.global_variables_initializer()

epochs = 1000
loss_trace = []
train_acc = []
test_acc = []

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(epochs):
        
        for step in range(len(x)//batch_size):
            X_batch, y_batch = next_batch(X_train, y_train, batch_size)
            feed = {X:X_batch, Y:y_batch}
        
            sess.run(train,feed_dict=feed)
          
        if (epoch % 50) == 0:
            Xte_batch, yte_batch = next_batch(X_train, y_train, batch_size)
            test_feed = {X:Xte_batch, Y: yte_batch}
            temp_loss = sess.run(loss, feed)
            temp_train_acc = sess.run(accuracy, feed)
            temp_test_acc = sess.run(accuracy, test_feed)
            loss_trace.append(temp_loss)
            train_acc.append(temp_train_acc)
            test_acc.append(temp_test_acc)
            print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch, temp_loss,
                                                                          temp_train_acc, temp_test_acc))

    y_pred = sess.run(pred,feed_dict = {X:X_test, Y: y_test})
    y_pred[y_pred==-1] = 0
    
            
# loss function
plt.plot(loss_trace)
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()  

# accuracy
plt.plot(train_acc, 'b-', label='train accuracy')
plt.plot(test_acc, 'k-', label='test accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Train and Test Accuracy')
plt.legend(loc='best')
plt.show()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
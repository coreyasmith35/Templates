# Random Forest using tensorflow

# Importing the libraries
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../DataSets/Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Placeholders
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

num_classes = 30

# Random Forest Hyperperamaters
perams = tensor_forest.ForestHParams(num_classes= num_classes, 
                                     num_features= 1, 
                                     num_trees= 10, 
                                     max_nodes= 10,
                                     regression = True).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(perams)

# Get training graph and loss
train = forest_graph.training_graph(X, Y)
loss = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer, _, _ = forest_graph.inference_graph(X)
prediction = tf.equal(tf.argmax(infer, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

init = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))
saver = tf.train.Saver()

# Session
with tf.Session() as sess:
    
    sess.run(init)
    
    prev_acc = 10000.0
    feed = {X:x, Y:y}
    
    epochs = 20000
    for i in range(epochs):
        
        sess.run(train,feed_dict=feed)
        
        if (i % 1000) == 0:
            acc = accuracy.eval({X:x, Y:y})
            print("Epoch: ", i, "\tAccuracy: ", acc)
        
            # Stop training when we reach a threshold
            if np.abs(prev_acc - acc) < 0.000001:
                break
        
            prev_acc = acc
     
    # Training done save a fitted line with size of x and step of 0.1
    X_grid = np.arange(min(x), max(x), 0.01)
    feed = {X:X_grid}
    y_pred = prediction.eval(feed)
    saver.save(sess,'./savedModels/savedModel.ckpt')
    print("Training Done! Training Accuracy=", acc)
    

# Plot
plt.plot(x, y, 'ro', label='Data')
plt.plot(X_grid, y_pred, label='Forest buckets')
plt.title('Random Forest w/ num_classes = %d' %num_classes)
plt.legend()
plt.show()


# Single pridiction on fitted line
single = 6.5
single = sc_X.transform(single).reshape(-1)

with tf.Session() as sess:
    
    saver.restore(sess,'./savedModels/savedModel.ckpt')
    
    single_pred = prediction.eval({X:single})
    
print('Single prediction at 6.5: ', sc_y.inverse_transform(single_pred)[0])

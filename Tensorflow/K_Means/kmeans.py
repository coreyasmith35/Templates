# K-Means Clustering

# Importing the libraries
import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../DataSets/Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values
y = dataset.iloc[:, 3].values

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

# Peramaters
num_features = 2
k = 5

# Placeholders
X = tf.placeholder(tf.float32, [None, num_features])
Y = tf.placeholder(tf.float32, [None, k]) # one outpu

kmeans = KMeans(inputs=X, num_clusters=k, 
                distance_metric='squared_euclidean',
                kmeans_plus_plus_num_retries=2)

# Build KMeans graph
training_graph = kmeans.training_graph()
(all_scores, cluster_idx, scores, _, kmeans_init, kmeans_training_op) = training_graph

# Remove tuple 
cluster_idx = cluster_idx[0]

# WCSS - Within Cluster Sum of Squares
wcss = tf.reduce_mean(tf.square(scores))

init = tf.global_variables_initializer()

d = []

epochs = 200
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    sess.run(kmeans_init, feed_dict={X: X_train})

    # Training cycle
    for epoch in range(epochs):
        
        _, temp_wcss, idx = sess.run([kmeans_training_op, wcss, cluster_idx], feed_dict={X:X_train})
        
        d.append(temp_wcss)
        
        if (epoch % 50) == 0:
            print("Step %i, WCSS: %f" % (epoch, temp_wcss))
            
    final_cluster_idx = idx

# loss function
plt.plot(d)
plt.title('Within Cluster Sum of Squares')
plt.xlabel('epoch')
plt.ylabel('wess')
plt.show()  

# Rescale the data
X_train = scaler_x.inverse_transform(X_train)
   
# Visualising the clusters
plt.scatter(X_train[final_cluster_idx == 0, 0], X_train[final_cluster_idx == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_train[final_cluster_idx == 1, 0], X_train[final_cluster_idx == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_train[final_cluster_idx == 2, 0], X_train[final_cluster_idx == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_train[final_cluster_idx == 3, 0], X_train[final_cluster_idx == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X_train[final_cluster_idx == 4, 0], X_train[final_cluster_idx == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

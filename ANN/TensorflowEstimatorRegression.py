# Tensorflow Estimator API

# Regression

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline

# Making some random data in the form:
# y = mx + b    with a little noise
x_data = np.linspace(0.0,10.0,1000000)
noise = np.random.randn(len(x_data))
y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data,columns=['X Data'])
y_df = pd.DataFrame(data=y_true,columns=['Y'])

my_data = pd.concat([x_df,y_df],axis = 1)


# plot the data
my_data.sample(n=250).plot(kind = 'scatter',x='X Data',y = 'Y')

# Feature Coloumns
feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]

# Making the Model
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

# Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_eval, y_train, y_eval = train_test_split(x_data,y_true,test_size=0.3)

# Input Fuctions
input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)
train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=8, num_epochs=1000, shuffle=False)
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval}, y_eval, batch_size=8, num_epochs=1000, shuffle=False)

# Train the model
estimator.train(input_func, steps=1000)

# Evaluate the Model
train_metrics = estimator.evaluate(input_fn=train_input_func, steps = 1000)
eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps = 1000)
print('Training Data Metrics')
print(train_metrics)
print('Eval Metrics')
print(eval_metrics)

# Given some x value what is its coraspoding y value according to the model
brand_new_data = np.linspace(0,10,10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data}, shuffle=False)
list(estimator.predict(input_fn=input_fn_predict))

predictions = []
for pred in estimator.predict(input_fn=input_fn_predict):
    predictions.append(pred['predictions'])

# Plot the model    
my_data.sample(n=250).plot(kind = 'scatter', x= 'X Data', y = 'Y')
plt.plot(brand_new_data,predictions, 'r')
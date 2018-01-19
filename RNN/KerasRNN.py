# Recurrent Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling - Normalisation
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timestamps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)  
    
# Reshaping   
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))   


# Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN   
regressor = Sequential()
  
# Adding the first LSTM layer and some Dropout regularisation
    # UNITS = number of cells/memory units
    # return sequences = set true if adding another layer after
    # input shape
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1 )))    
regressor.add(Dropout(.2))    
    
# Adding the second LSTM layer and some Dropout regularisation    
regressor.add(LSTM(units = 50, return_sequences = True))    
regressor.add(Dropout(.2))    
    
# Adding the third LSTM layer and some Dropout regularisation      
regressor.add(LSTM(units = 50, return_sequences = True))    
regressor.add(Dropout(.2))     
    
# Adding the fourth/last LSTM layer and some Dropout regularisation      
regressor.add(LSTM(units = 50))    
regressor.add(Dropout(.2))     

# Adding the output layer
    # units = the output node(predicting stock price at t)
regressor.add(Dense(units = 1))

# Compiling the RNN
    # Recommend optimizer = 'RMSprops' in general for RNNs
    # mean squared error for RNNs
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train,
              y_train,
              epochs = 100,
              batch_size = 32)

# Getting the real stock price of January 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Making the prediction of January 2017
    # 0 - Vertical concat 
    # 1 - Horazontal concat
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
    # removing unneeded values
inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60 : ].values
    # Get the right np shape
inputs = inputs.reshape(-1,1)
    # Scale inputs by same scaling as trained data
inputs = sc.transform(inputs) 
    # Creating a data structure with 60 timestamps and 1 output
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
 
# Makking it into the 3D format; Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Making the prediction
predicted_stock_price = regressor.predict(X_test)

# Inverse the scaling
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
 
# Visualising the results 
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


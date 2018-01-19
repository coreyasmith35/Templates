# AutoEncoder pytorch

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('data/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('data/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network

# Peramaters tunable
numFirstLayer = 20
numSecondLayer = 10
learningRate = 0.01
weightDecay = 0.5
activationType = nn.Sigmoid()
criterion = nn.MSELoss()
numEpoch = 1000


# input -> 20 -> 10 -> 20 -> output
class SAE(nn.Module):
    def __init__(self, numFirstLayer, numSecondLayer, activationType, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, numFirstLayer)
        self.fc2 = nn.Linear(numFirstLayer, numSecondLayer)
        self.fc3 = nn.Linear(numSecondLayer, numFirstLayer)
        self.fc4 = nn.Linear(numFirstLayer, nb_movies)
        self.activation = activationType

    def forward(self, x):
        x = self.activation(self.fc1(x)) # incode
        x = self.activation(self.fc2(x)) # incode
        x = self.activation(self.fc3(x)) # decode
        x = self.fc4(x) # decode
        return x
    
    
sae = SAE(numFirstLayer, numSecondLayer, activationType)
optimizer = optim.RMSprop(sae.parameters(), lr = learningRate, weight_decay = weightDecay)

# Training the SAE
for epoch in range(1, numEpoch + 1):
    train_loss = 0
    count = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) >0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0] * mean_corrector)
            count += 1.
            optimizer.step()
    print('Epoch: ' + str(epoch) + ' Loss: ' + str(train_loss/count))

# Test set Error
test_loss = 0
count = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) >0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0] * mean_corrector)
        count += 1.
print('Test Loss: ' + str(test_loss/count))



































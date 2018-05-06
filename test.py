import numpy as np
import matplotlib.pyplot as plt
import pandas
import math,os
import torch
from torch import autograd
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

# Here the input features are two dimension unlike in RNN_power_predict_temp,
# where input is one dimension feature vector
df = pandas.read_csv('practise_data//household_power_consumption', sep=';',na_values=['?']) # convert missing value -> ? to NAN

df = df.dropna()# Remove 

limit_rows = 90000000

input_size = 1 # as input size is Nx1 (one feature per input)
hidden_size = 2
num_layers = 1
learning_rate = 0.003
limit_rows = 2000
sequence = 5
sequence = 100
batch_size = 16


df = df[:limit_rows]
# Get the header info
df.columns.values.tolist()
# get global active power,vlotage,intensity
data = df[['Global_active_power','Voltage','Global_intensity']]

data.info()

temp = data[['Global_active_power','Voltage']]

power = list(temp['Global_active_power'].get_values().flatten())
voltage = list(temp['Voltage'].get_values().flatten())


result = []
for index in range(len(power) - sequence):
    # get the next n sequence data
    temp = []
    for data in power[index: index + sequence]:
        temp.append(data)
    result.append(temp)
    

result = np.array(result)  # shape (2049230, 50)

row = int(round(0.9 * result.shape[0]))
train = result[:row, :]
np.random.shuffle(train)
X_train = train[:, :-1]
y_train = train[:, -1]
X_test = result[row:, :-1]
y_test = result[row:, -1]


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))# as dimenion is 1 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
train_loader_x = torch.utils.data.DataLoader(dataset=X_train, 
                                           batch_size=batch_size, 
                                           shuffle=False)

train_loader_y = torch.utils.data.DataLoader(dataset=y_train, 
                                           batch_size=batch_size, 
                                           shuffle=False)

test_loader_x = torch.utils.data.DataLoader(dataset=X_test, 
                                          batch_size=batch_size, 
                                          shuffle=False)

test_loader_y = torch.utils.data.DataLoader(dataset=y_test, 
                                          batch_size=batch_size, 
                                          shuffle=False)

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2,1)  # 2 for bidirection  and 1 as output is one
    
    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)) # 2 for bidirection 
        c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out


#  (#examples, #values in sequences, dim. of each value).
rnn = BiRNN(input_size, hidden_size, num_layers)
rnn.cuda()
# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

for epoch in range(100):
    for data in zip(train_loader_x,train_loader_y): 
         #Forward + Backward + Optimize
        optimizer.zero_grad()
        #X_train_ = np.reshape(data[0], (data[0].shape[0], data[0].shape[1], 1))# as dimenion is 2 
        X_train_ = Variable(data[0].float().cuda())
        temp = data[1].float().cuda()
        labels = Variable(temp.view(temp.numel(),1))
        outputs = rnn(X_train_)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print("Loss is {}".format(loss.data[0]))

import numpy as np
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn
import torch
from template_AI.network_collection import RNN

from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()

hidden_size = 5
input_size = 1

rnn_input_size = input_size
rnn = RNN(rnn_input_size, hidden_size, 1).to(device)
optimizer = optim.Adam(rnn.parameters(), lr=0.00001)


rnn = rnn.float()

input_data = [[0.8], [0.1], [0.3], [0.1], [0.5], [0.2]]
input_data = torch.from_numpy(np.array(input_data)).float()

f_data = [[0.0], [0.8], [0.1], [0.3], [0.1], [0.5]]
f_data = torch.from_numpy(np.array(f_data)).float()
hidden = torch.from_numpy(np.zeros(hidden_size)).float()

for i in range(1000):
    for d in zip(input_data, f_data):
        x, hidden = rnn.forward(deepcopy(d[0]), hidden)
        loss = criterion(x, deepcopy(d[1]))
        optimizer.zero_grad()
        loss.backward()
        print("first?")
        optimizer.step()

    print(loss)
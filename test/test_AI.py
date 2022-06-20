import torch
from torch import nn
import numpy as np


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2output = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), 1)
        hidden = torch.sigmoid(self.in2hidden(combined))
        output = self.in2output(combined)
        return output, hidden

    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))

hidden_size = 16
learning_rate = 0.0001

model = MyRNN(16, hidden_size, 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 2
print_interval = 10

from world import GH


for epoch in range(num_epochs):

    for i in range(10000):
        hidden_state = model.init_hidden()
        gh = GH()
        gh.get_world()
        loss = torch.tensor(0.0)
        for char in range(1000):
            flatten_list = [sum(gh.get_world(), [])]
            output, hidden_state = model(torch.FloatTensor(flatten_list), hidden_state)

            loss += criterion(output, torch.FloatTensor([gh.get_correct()]))
            gh.next_frame()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        if (i + 1) % print_interval == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{i + 1}/{10000}], "
                f"Loss: {loss.item():.4f}"
            )
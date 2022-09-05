import torch
from torch import nn

import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
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


class Encoder(nn.Module):
    def __init__(self, width, height, num_filters_layer_one, num_filters_layer_two, latent_size):
        super(Encoder, self).__init__()

        self.c1 = nn.Conv2d(3, num_filters_layer_one, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.c2 = nn.Conv2d(num_filters_layer_one, num_filters_layer_two, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.c_lin = nn.Linear(width * height * num_filters_layer_two / 4, latent_size)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c_lin(x))
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size*2),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size*2, input_size*4),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size*4, input_size*8),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size*8, output_size*4),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size*4, output_size),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.encoder(x)


class FC(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FC, self).__init__()
        # Our first linear layer take input_size, in this case 784 nodes to 50
        # and our second linear layer takes 50 to the num_classes we have, in
        # this case 10.
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        """
        x here is the mnist images and we run it through fc1, fc2 that we created above.
        we also add a ReLU activation function in between and for that (since it has no parameters)
        I recommend using nn.functional (F)
        """

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

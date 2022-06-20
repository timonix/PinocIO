# Imports
import json
import os

import torch
#import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
#import torchvision.datasets as datasets  # Standard datasets
#import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!
from os import listdir
import numpy as np
from PIL import Image

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
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

# Hyperparameters of our neural network which depends on the dataset, and
# also just experimenting to see what works well (learning rate for example).
input_size = 200*300*3
num_classes = 3
learning_rate = 0.001
batch_size = 64
num_epochs = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Set device cuda for GPU if it's available otherwise run on the CPU


    print("Preparing train dataset")

    # Load Training and Test data
    train_dataset = []
    facit_dict = dict()
    with open("facit.json", "r") as facit:
        facit_dict = json.load(facit)
        for path, values in tqdm(facit_dict.items()):
            img = Image.open(path)
            train_dataset.append(np.asarray(img))


    #test_dataset =
    #train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    print("Initializing network")

    # Initialize network
    model = NN(input_size=input_size, num_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Training network")

    # Train Network
    for epoch in range(num_epochs):
        for i, image in enumerate(tqdm(train_dataset)):

            # TODO: Chunks!!!

            # forward
            output = model(torch.FloatTensor(image.flatten()))
            loss = criterion(output, torch.FloatTensor(facit_dict["Images/image"+str(i)+".png"]))

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            #if i > 1000:
            #    break

        # Moved this here, not tested
        torch.save(model.state_dict(), "model.m")


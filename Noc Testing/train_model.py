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
        self.fc1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc_lin = nn.Linear(input_size_linear, 3)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = torch.reshape(x, (x.shape[0],-1))
        x = self.fc_lin(x)
        return x

# Hyperparameters of our neural network which depends on the dataset, and
# also just experimenting to see what works well (learning rate for example).
input_size = 3*3*3 # 200*300*3 for old pixel-based linear network
input_size_linear = 200*300*64
num_classes = 3
learning_rate = 0.001
batch_size = 64
num_epochs = 3
num_chunks = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Set device cuda for GPU if it's available otherwise run on the CPU


    #print("Preparing train dataset")

    # Load Training and Test data
    train_dataset = []
    facit_dict = dict()
    with open("facit.json", "r") as facit:
        facit_dict = json.load(facit)
        for path, values in tqdm(facit_dict.items(), desc="Preparing train dataset"):
            img = Image.open(path)
            arr = np.asarray(img)
            arr = np.moveaxis(arr, (2), (0))
            train_dataset.append(arr)

    train_dataset_total = np.stack(train_dataset)
    chunks = np.array_split(train_dataset_total, num_chunks)



    #test_dataset =
    #train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    print("Initializing network")

    # Initialize network
    model = NN(input_size=input_size, num_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Preparing Facit")

    facit_arr = []
    for i, image in enumerate(tqdm(facit_dict)):
        facit_arr.append(facit_dict["Train Images/image" + str(i) + ".png"])

    facit = np.stack(facit_arr)
    facit_chunks = np.array_split(facit, num_chunks)

    print("Training network")

    try:
        model.load_state_dict(torch.load("model.m"))
    except:
        pass

    # Train Network
    for epoch in range(num_epochs):
        for i, chunk in enumerate(tqdm(chunks)):

            # forward
            output = model(torch.FloatTensor(chunk))
            loss = criterion(output, torch.FloatTensor(facit_chunks[i]))

            # backward
            optimizer.zero_grad()
            loss.backward()

            if (i + 1) % 5 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{i + 1}/{10000}], "
                    f"Loss: {loss.item():.4f}"
                )

            # gradient descent or adam step
            optimizer.step()

            #if i > 1000:
            #    break

            # Moved this here, not tested
        torch.save(model.state_dict(), "model.m")


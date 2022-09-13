import json

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules

from template_AI.network_collection import Encoder
from template_AI.network_collection import Decoder


latent_size = 100
height = 200
width = 300
learning_rate = 0.001
batch_size = 64
num_epochs = 3
num_chunks = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(width, height, 32, 32, latent_size).to(device)
decoder = Decoder(latent_size, height*width).to(device)

print(device)

if __name__ == "__main__":
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

    criterion = nn.MSELoss()
    optimizer_en = optim.Adam(encoder.parameters(), lr=learning_rate)
    optimizer_de = optim.Adam(decoder.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, chunk in enumerate(tqdm(chunks)):

            # forward
            latent_space = encoder(torch.FloatTensor(chunk))
            output = decoder(latent_space)

            loss = criterion(output, chunk)

            # backward
            optimizer_de.zero_grad()
            optimizer_en.zero_grad()
            loss.backward()

            if (i + 1) % 5 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{i + 1}/{10000}], "
                    f"Loss: {loss.item():.4f}"
                )

            # gradient descent or adam step
            optimizer_en.step()
            optimizer_de.step()

            #if i > 1000:
            #    break

            # Moved this here, not tested
        torch.save(encoder.state_dict(), "encoder.m")
        torch.save(decoder.state_dict(), "decoder.m")
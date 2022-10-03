import cv2
import torch

import numpy as np
from PIL import Image

import time

from tqdm import tqdm
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn

from os.path import exists

from template_AI.network_collection import Encoder64 as Encoder
from template_AI.network_collection import Decoder64 as Decoder
from Data import Data

image_channels = 3
encoder_base_size = 32
decoder_base_size = 32
latent_dim = 100

bath_size = 2500

learning_rate = 0.00001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(image_channels, latent_dim, layer_params=[80, 72, 64, 56, 48, 40, 80]).to(device)
decoder = Decoder(image_channels, latent_dim, layer_params=[80, 40, 48, 56, 64, 72, 80]).to(device)

criterion = nn.MSELoss()

if exists("encoder.m"):
    encoder.load_state_dict(torch.load("encoder.m"))
    decoder.load_state_dict(torch.load("decoder.m"))

    encoder.eval()
    decoder.eval()

if __name__ == "__main__":
    print("Loading data")
    dataLoader = Data(path="C:\\Users\\u057742.CORP\\Train Images\\")
    optimizer_en = optim.Adam(encoder.parameters(), lr=learning_rate)
    optimizer_de = optim.Adam(decoder.parameters(), lr=learning_rate)
    print("shuffle data")

    dataLoader.shuffle()
    data = dataLoader.getNext(batch_size=bath_size)
    print("Started")
    while data is not None:
        x = np.array(data)
        x = x / 255
        x = np.swapaxes(x, 1, 3)

        x = torch.FloatTensor(x).to(device)

        facit = x

        print(x.shape)

        data = dataLoader.getNext(batch_size=bath_size)

        for epoch in tqdm(range(501)):
            latent_space = encoder(x)
            output = decoder(latent_space)

            loss = criterion(output, facit)

            if epoch % 500 == 0:
                print(
                    f"Loss: {loss.item():.8f}"
                )

                torch.save(encoder.state_dict(), "encoder.m")
                torch.save(decoder.state_dict(), "decoder.m")

            optimizer_de.zero_grad()
            optimizer_en.zero_grad()
            loss.backward()

            optimizer_en.step()
            optimizer_de.step()

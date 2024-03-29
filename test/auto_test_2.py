import torch

import numpy as np
from PIL import Image

import time

from tqdm import tqdm
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn

from template_AI.network_collection import Encoder_B as Encoder
from template_AI.network_collection import Decoder_B as Decoder
from image_generator import generate_train_dataset

import glob


filelist = glob.glob('../Train Images/*.png')

#print(filelist)

image_channels = 3
encoder_base_size = 32
decoder_base_size = 32
latent_dim = 10

learning_rate = 0.00001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(image_channels, encoder_base_size, latent_dim).to(device)
decoder = Decoder(image_channels, decoder_base_size, latent_dim).to(device)

criterion = nn.MSELoss()

if __name__ == "__main__":

    # x = np.array([np.array(Image.open(fname)) for fname in filelist])
    # x = np.swapaxes(x, 1, 3)
    # x = x[-1:]

    # x = torch.FloatTensor(x)

    # latent_space = encoder.forward(x)
    # decoder(latent_space)

    optimizer_en = optim.Adam(encoder.parameters(), lr=learning_rate)
    optimizer_de = optim.Adam(decoder.parameters(), lr=learning_rate)

    for iteration in range(10):

        generate_train_dataset(1000, 32, 32)

        x = np.array([np.array(Image.open(fname)) for fname in filelist])
        x = x/255

        x = np.swapaxes(x, 1, 3)

        x = torch.FloatTensor(x).to(device)




        facit = x


        for epoch in tqdm(range(3000)):
            # forward

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

            pass

    image = output[0]*255
    image = image.cpu().detach().numpy()
    for i in image.tolist():
        print(i)
    image = np.swapaxes(image, 0, 2)
    image = image.astype("uint8")
    image = image + 11

    Image.fromarray(image, 'RGB').show()
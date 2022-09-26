import torch

import numpy as np
from PIL import Image

import time

from tqdm import tqdm
from tqdm import trange
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn

from template_AI.network_collection import Encoder_B as Encoder
from template_AI.network_collection import Decoder_B as Decoder
from image_generator import generate_train_dataset

import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


filelist = glob.glob('../Train Images/*.png')
test_files = glob.glob('../Test Images/')

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

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True


        generate_train_dataset(1000, 32, 32)

        x = np.array([np.array(Image.open(fname)) for fname in filelist])
        x = x/255


        x = np.swapaxes(x, 1, 3)

        x = torch.FloatTensor(x).to(device)

        if epoch % 1 == 0:
            #print(
            #    # f"Epoch [{epoch + 1}/{100}], "
            #    f"Loss: {loss.item():.4f}"
            #)
            torch.save(encoder.state_dict(), "encoder.m")
            torch.save(decoder.state_dict(), "decoder.m")

        facit = x

        for epoch in tqdm(range(3000)):
            # forward

            latent_space = encoder(x)
            output = decoder(latent_space)


        torch.save(encoder.state_dict(), "encoder.m")
        torch.save(decoder.state_dict(), "decoder.m")

        if epoch % 500 == 0:
            print(
                f"Loss: {loss.item():.8f}"
            )

            torch.save(encoder.state_dict(), "encoder.m")
            torch.save(decoder.state_dict(), "decoder.m")


    image = output[0] * 255
    image = image.detach().numpy()
    image = np.swapaxes(image, 0, 2)
    image = image.astype(np.int8)
    Image.fromarray(image, 'RGB').show()


            optimizer_en.step()
            optimizer_de.step()


    pass

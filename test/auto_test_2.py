import torch

import numpy as np
from PIL import Image

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
latent_dim = 300

image_width = 32
image_height = 32

learning_rate = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(image_channels, encoder_base_size, latent_dim, 60800).to(device)
decoder = Decoder(image_channels, decoder_base_size, latent_dim, 300*200).to(device)

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

        generate_train_dataset(10000, 32, 32)

        x = np.array([np.array(Image.open(fname)) for fname in filelist])
        x = np.swapaxes(x, 1, 3)

        x = torch.FloatTensor(x)
        facit = x.reshape(x.shape[0], 200*300*3)

        for epoch in tqdm(range(100), desc="Training Encoder"):    # Training the encoder, decoder static
            # forward

            latent_space = encoder(x)
            output = decoder(latent_space)

            loss = criterion(output, facit)

            if epoch % 1 == 0:
                print(
                    # f"Epoch [{epoch + 1}/{100}], "
                    f"Loss: {loss.item():.4f}"
                )
                torch.save(encoder.state_dict(), "encoder.m")
                # torch.save(decoder.state_dict(), "decoder.m")

            optimizer_de.zero_grad()
            optimizer_en.zero_grad()
            loss.backward()

            optimizer_en.step()
            # optimizer_de.step()

        for epoch in tqdm(range(100), desc="Training Decoder"):  # Training the decoder, encoder static
            # forward

            latent_space = encoder(x)
            output = decoder(latent_space)

            loss = criterion(output, facit)

            if epoch % 1 == 0:
                print(
                    # f"Epoch [{epoch + 1}/{100}], "
                    f"Loss: {loss.item():.4f}"
                )
                # torch.save(encoder.state_dict(), "encoder.m")
                torch.save(decoder.state_dict(), "decoder.m")

            optimizer_de.zero_grad()
            optimizer_en.zero_grad()
            loss.backward()

            # optimizer_en.step()
            optimizer_de.step()

            pass

    image = output[0].reshape(200, 300, 3)
    Image.fromarray(image.detach().numpy(), 'RGB').show()
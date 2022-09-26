import torch

import numpy as np
from PIL import Image
import cv2 as cv
import time

from tqdm import tqdm
from tqdm import trange
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn

from template_AI.network_collection import Encoder_B as Encoder
from template_AI.network_collection import Decoder_B as Decoder
import webcam_shenanigans as webcam

import matplotlib.pyplot as plt


image_channels = 3
encoder_base_size = 32
decoder_base_size = 32
latent_dim = 300

image_width = 32
image_height = 32

learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(image_channels, encoder_base_size, latent_dim, 16384).to(device)
decoder = Decoder(image_channels, decoder_base_size, latent_dim).to(device)

criterion = nn.MSELoss()

if __name__ == '__main__':

    optimizer_en = optim.Adam(encoder.parameters(), lr=learning_rate)
    optimizer_de = optim.Adam(decoder.parameters(), lr=learning_rate)

    encoder.load_state_dict(torch.load("encoder.m"))
    encoder.eval()

    decoder.load_state_dict(torch.load("decoder.m"))
    decoder.eval()

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    loss_arr = []

    train = True

    frames = webcam.video_to_arr('../Videos/long_vid.mp4')

    while True:

        #frame = webcam.capture_frame()


        #original_image = frame
        #cv.imshow('Original Image', original_image)

        #if cv.waitKey(1) == ord('q'):
        #    webcam.disconnect()
        #    break

        #original_dimensions = (frame.shape[1], frame.shape[0])

        #dim = (128, 128)
        #frame = [cv.resize(frame, dim, interpolation=cv.INTER_AREA)]

        x = np.array(frames)
        x = x / 255
        x = np.swapaxes(x, 1, 3)

        x = torch.FloatTensor(x)
        facit = x

        if train:
            for epoch in tqdm(range(100), desc="Training Encoder and Decoder"):
                # forward

                latent_space = encoder(x)
                output = decoder(latent_space)

                loss = criterion(output, facit)

                loss_arr.append(loss.item())

                optimizer_de.zero_grad()
                optimizer_en.zero_grad()
                loss.backward()

                optimizer_en.step()
                optimizer_de.step()

            torch.save(encoder.state_dict(), "encoder.m")
            torch.save(decoder.state_dict(), "decoder.m")

            #encoder._save_to_state_dict(torch.save("encoder.m"))
            #decoder._save_to_state_dict(torch.save("decoder.m"))

        else:
            latent_space = encoder(x)
            output = decoder(latent_space)

        #plt.plot(loss_arr)
        #plt.show()

        image = output[0] * 255
        image = image.detach().numpy()
        image = np.swapaxes(image, 0, 2)
        image = image.astype(np.uint8)
        #image = Image.fromarray(image, 'RGB')
        #image = cv.fromarray(image)

        dim = (640, 480)
        image = cv.resize(image, dim, interpolation=cv.INTER_AREA)
        cv.imshow('Result', image)

        if cv.waitKey(1) == ord('q'):
            webcam.disconnect()
            break

        pass




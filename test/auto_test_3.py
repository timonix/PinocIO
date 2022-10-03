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
from image_generator import generate_train_dataset

import glob

from timeit import default_timer as timer



#print(filelist)
from cunker import cunker, decunker

image_channels = 3
encoder_base_size = 32
decoder_base_size = 32
latent_dim = 100

learning_rate = 0.00001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(image_channels, latent_dim, layer_params=[80, 72, 64, 56, 48, 40, 32]).to(device)
decoder = Decoder(image_channels, latent_dim, layer_params=[32, 40, 48, 56, 64, 72, 80]).to(device)

if exists("encoder.m"):
    encoder.load_state_dict(torch.load("encoder.m"))
    decoder.load_state_dict(torch.load("decoder.m"))

    encoder.eval()
    decoder.eval()

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

    #encoder.load_state_dict(torch.load("encoder.m"))
    #decoder.load_state_dict(torch.load("decoder.m"))

    #encoder.eval()
    #decoder.eval()

    cam = cv2.VideoCapture(0)
    for iteration in range(1000000):

        ret_val, cv_img = cam.read()

        color_coverted = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)

        img = pil_image

        image_size = (640, 480)
        cunk_size = 64

        cc = cunker(cunk_size)
        ll = []
        for c in cc.cunk(img):
            ll.append(np.array(c))
        x = np.array(ll)

        x = x/255


        x = np.swapaxes(x, 1, 3)

        x = torch.FloatTensor(x).to(device)


        facit = x


        for epoch in tqdm(range(50)):
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





        image = output.cpu().detach().numpy()
        image = np.swapaxes(image, 1, 3)
        image = image*255
        image = image.astype("uint8")
        pila = []
        for im in image:
            pila.append(Image.fromarray(im))

        dc = decunker(image_size, 64)
        dc.decunk(pila).show()



# import libraries
from vidgear.gears import CamGear
from cunker import cunker, decunker

from tqdm import tqdm
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn

import sys, os
sys.path.append('/workspace/PinocIO')

from template_AI.network_collection import Encoder64 as Encoder
from template_AI.network_collection import Decoder64 as Decoder
import cv2

import cv2
import torch

import numpy as np
from PIL import Image

from os.path import exists

image_channels = 3
encoder_base_size = 32
decoder_base_size = 32
latent_dim = 10

learning_rate = 0.00001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(image_channels, latent_dim).to(device)
decoder = Decoder(image_channels, latent_dim).to(device)


if exists("encoder.m"):
    encoder.load_state_dict(torch.load("encoder.m"))
    decoder.load_state_dict(torch.load("decoder.m"))

    encoder.eval()
    decoder.eval()

criterion = nn.MSELoss()


stream = CamGear(source='https://www.youtube.com/watch?v=uySgklnlX3Y', stream_mode=True,
                 logging=True).start()  # YouTube Video URL as input

optimizer_en = optim.Adam(encoder.parameters(), lr=learning_rate)
optimizer_de = optim.Adam(decoder.parameters(), lr=learning_rate)

# infinite loop
while True:

    ll = []

    for i in range(10):
        frame = stream.read()
        # read frames

        # check if frame is None
        if frame is None:
            print("NO MORE VIDEO")
            exit()

        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)

        img = pil_image
        image_size = (frame.shape[1], frame.shape[0])
        cunk_size = 64

        cc = cunker(cunk_size)
        for c in cc.cunk(img):
            ll.append(np.array(c))



    x = np.array(ll)

    x = x / 255

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
    image = image * 255
    image = image.astype("uint8")
    pila = []
    for im in image:
        pila.append(Image.fromarray(im))

    dc = decunker(image_size, 64)
    # dc.decunk(pila).show()


    # do something with frame here

    # Show output window

# close output window

# safely close video stream.
stream.stop()
import torch

import numpy as np
from PIL import Image

from template_AI.network_collection import Encoder_B as Encoder
from template_AI.network_collection import Decoder_B as Decoder

import glob

filelist = glob.glob('../Noc Testing/Train Images/*.png')

print(filelist)

image_channels = 3
encoder_base_size = 32
decoder_base_size = 32
latent_dim = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(image_channels, encoder_base_size, latent_dim).to(device)
decoder = Decoder(image_channels, decoder_base_size, latent_dim).to(device)

if __name__ == "__main__":
    x = np.array([np.array(Image.open(fname)) for fname in filelist])
    x = np.swapaxes(x, 1, 3)
    x = torch.FloatTensor(x)
    encoder.forward(x)
    pass
import os
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageOps
from torchvision import transforms
from pokenet import encoder, decoder
import pokenet
from torch import nn
from tqdm import tqdm

from pathlib import Path
from natsort import natsorted, ns

import settings

import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader

convert_tensor = transforms.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.MSELoss()


class PokeImages(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        p = Path(root_dir)
        all_sessions = [x for x in p.iterdir() if x.is_dir()]

        self.files = []

        for session in all_sessions:
            session_path = os.path.join(session, Path("images/"))
            session_files = [join(session_path, f) for f in listdir(session_path) if
                             f.endswith(".png") and isfile(join(session_path, f))]
            session_files = natsorted(session_files, key=lambda y: y.lower())
            self.files = self.files + session_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        image = convert_tensor(image)
        return image


num_epochs = 50
subEpochs = 5
if __name__ == "__main__":
    print("Loading data00")
    data = PokeImages(settings.DATA_PATH)
    training_generator = DataLoader(data, batch_size=1000, shuffle=True)

    optimizer_en = optim.Adam(encoder.parameters(), lr=0.0001)
    optimizer_de = optim.Adam(decoder.parameters(), lr=0.0001)

    print("Started")

    for epoch in range(num_epochs):
        for batch in tqdm(training_generator):
            batch = batch.to(device)
            for i in range(subEpochs):
                latent_space = encoder(batch)
                output = decoder(latent_space)

                loss = criterion(output, batch)

                optimizer_de.zero_grad()
                optimizer_en.zero_grad()
                loss.backward()

                optimizer_en.step()
                optimizer_de.step()

        print(f"Loss: {loss.item():.8f}")
        pokenet.save_decoder()
        pokenet.save_encoder()

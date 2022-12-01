import pickle

import numpy as np
import retro
import torch
import random
import re

import glob, os


from english_words import english_words_set


from PIL import Image
from cunker import cunker, decunker

from os.path import exists

from template_AI.network_collection import Encoder64 as Encoder
from template_AI.network_collection import Decoder256 as Decoder
from template_AI.network_collection import RNN as Action
from template_AI.network_collection import FC as RewNet

from tqdm import tqdm
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn

import mgzip

folder = "../data/"

image_channels = 3
latent_dim = 10
world_state_size = 200
action_space = 12

learning_rate = 0.00001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Networks
encoder = Encoder(image_channels, latent_dim).to(device)
decoder = Decoder(image_channels, world_state_size + action_space).to(device)
action_net = Action(world_state_size, world_state_size, action_space).to(device)
reward_net = RewNet(world_state_size + action_space, 1).to(device)

optimizer_encoder = optim.Adam(encoder.parameters(), lr=learning_rate)
optimizer_decoder = optim.Adam(decoder.parameters(), lr=learning_rate)
optimizer_action_net = optim.Adam(action_net.parameters(), lr=learning_rate)
optimizer_reward_net = optim.Adam(reward_net.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

if exists("gym_encoder.m"):
    encoder.load_state_dict(torch.load("gym_encoder.m"))
    encoder.eval()

if exists("gym_decoder.m"):
    decoder.load_state_dict(torch.load("gym_decoder.m"))
    decoder.eval()

print(action_net)

def main():
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    active_session = random.choice(subfolders)
    print(active_session)
    print(active_session+"/main*.dat")
    world_state = torch.FloatTensor(np.zeros(200)).to(device)
    for file in glob.glob(active_session+"/main*.dat"):
        frame = re.findall(r'\d+', file)[0]

        with mgzip.open(file, 'rb') as f:
            data = pickle.load(f)
        img_file = glob.glob(active_session + "/image*"+frame+".dat")[0]
        with mgzip.open(img_file, 'rb') as f:
            image = pickle.load(f)

        latent_space = encoder(data).reshape((200))
        act, world_state = action_net.forward(latent_space, hidden_state=world_state)
        decoder_input = torch.hstack((world_state, act))
        decoder_input = torch.reshape(decoder_input, (1, 212))
        imagination = decoder(decoder_input)

        loss = criterion(image, imagination)

        estimated_reward = reward_net.forward(torch.hstack((world_state, act)))
        rr = torch.tensor([1000], dtype=float)
        loss2 = criterion(estimated_reward, rr)

        optimizer_decoder.zero_grad()
        optimizer_encoder.zero_grad()
        optimizer_action_net.zero_grad()
        optimizer_reward_net.zero_grad()
        loss.backward()
        loss2.backward()

        optimizer_encoder.step()
        optimizer_decoder.step()
        optimizer_action_net.step()


if __name__ == "__main__":
    main()

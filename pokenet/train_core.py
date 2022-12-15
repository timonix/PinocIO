import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
from os.path import isfile, join
from os import listdir
from natsort import natsorted, ns
from PIL import Image, ImageOps
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import random

import pickle


from pokenet import device, ACTIONS, WORLD_SIZE, IMAGE_SIZE, REWARD_SIZE, encoder, core_net, save_core

core_net_optimizer = Adam(core_net.parameters(), lr=0.0001, weight_decay=0.0001)

import settings

convert_tensor = transforms.ToTensor()



class Session(Dataset):
    def __init__(self, root_dir, session_id):
        self.dir = root_dir + "/" + session_id

        session_images_path = os.path.join(self.dir, Path("images/"))
        session_images = [join(session_images_path, f) for f in listdir(session_images_path) if
                          f.endswith(".png") and isfile(join(session_images_path, f))]
        session_images = natsorted(session_images, key=lambda y: y.lower())
        self.files = session_images

        file = open(self.dir + "/" + session_id + ".pact", 'rb')
        self.actions = pickle.load(file)
        file.close()
        file = open(self.dir + "/" + session_id + ".prew", 'rb')
        self.rewards = pickle.load(file)
        file.close()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        image = convert_tensor(image)
        return image, self.rewards[idx], self.actions[idx]


def all_sessions():
    listdir = []
    for file in os.listdir(settings.DATA_PATH):
        d = os.path.join(settings.DATA_PATH, file)
        if os.path.isdir(d):
            listdir.append(d)

    return random.shuffle(listdir)

action_matrix = torch.eye(len(ACTIONS)).to(device)

item = None
next_item = None
loss_fn = nn.MSELoss()

multi_sess = []
for i in range(50):
    multi_sess += all_sessions()
for session_name in tqdm(multi_sess):
    world = torch.zeros((1, WORLD_SIZE)).to(device)
    item = None
    next_item = None
    loss = None
    core_net_optimizer.zero_grad()
    sess = Session(settings.DATA_PATH, session_name)
    for i in tqdm(sess):
        item = next_item
        next_item = i
        if item is None:
            continue

        tensor_image = item[0].to(device) / 255.0
        tensor_image = torch.reshape(tensor_image, (1, tensor_image.shape[0], tensor_image.shape[1], tensor_image.shape[2]))
        latent = encoder(tensor_image).detach()
        action = action_matrix[item[2]]
        action = torch.reshape(action, (1, action.shape[0]))

        core_input = torch.cat((world, latent, action), dim=1).float()
        core = core_net(core_input)
        world, predicted_latent, predicted_reward = core.split([WORLD_SIZE, IMAGE_SIZE, REWARD_SIZE], dim=1)

        next_image = next_item[0].to(device) / 255.0
        next_image = torch.reshape(next_image, (1, next_image.shape[0], next_image.shape[1], next_image.shape[2]))
        next_latent = encoder(next_image).detach()

        actual_reward = torch.tensor([[item[1]]]).to(device)

        step_facit = torch.cat((next_latent, actual_reward), dim=1).float()

        actual = torch.cat((predicted_latent, predicted_reward), dim=1).float()

        if loss:
            loss += loss_fn(actual, step_facit)
        else:
            loss = loss_fn(actual, step_facit)

    loss.backward()
    core_net_optimizer.step()
    print(loss)
    save_core()



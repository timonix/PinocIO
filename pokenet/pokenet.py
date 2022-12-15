from pyboy import PyBoy
from pyboy import WindowEvent
import torch
import torch.nn as nn

from os.path import exists
import settings

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = [(WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
           (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
           (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
           (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
           (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
           (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
           (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START)
           ]

NUM_ACTIONS = len(ACTIONS)
WORLD_SIZE = 1000
IMAGE_SIZE = 1000
LATENT_SIZE = IMAGE_SIZE
HIDDEN_SIZE = 1000
REWARD_SIZE = 1

MAGIC = (9, 10)
MAGIC_NUMBER = MAGIC[0] * MAGIC[1]

core_net = nn.Sequential(nn.Linear(NUM_ACTIONS + WORLD_SIZE + IMAGE_SIZE, HIDDEN_SIZE*2),
                         nn.ReLU(),
                         nn.Linear(HIDDEN_SIZE*2, HIDDEN_SIZE),
                         nn.ReLU(),
                         nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                         nn.ReLU(),
                         nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                         nn.ReLU(),
                         nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE*2),
                         nn.ReLU(),
                         nn.Linear(HIDDEN_SIZE*2, WORLD_SIZE + REWARD_SIZE + IMAGE_SIZE),
                         nn.Sigmoid()).to(device)

encoder = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),  # 128x128 => 64x64
    nn.ReLU(),  # [32, 80, 72]
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.ReLU(),  #
    nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),  # 64x64 => 32x32
    nn.ReLU(),  # [32, 40, 36]
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.ReLU(),  #
    nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
    nn.ReLU(),  # [32, 20, 18]
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.ReLU(),  #
    nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
    nn.ReLU(),  # [64, 10, 9]
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Flatten(1, -1),  # Image grid to single feature vector [64, 25]
    nn.Linear(64 * 10 * 9, LATENT_SIZE)
).to(device)

decoder = nn.Sequential(  # large to small
    nn.Linear(LATENT_SIZE, MAGIC_NUMBER * 64),
    nn.Unflatten(1, (64, MAGIC[0], MAGIC[1])),
    nn.ConvTranspose2d(64, 64, kernel_size=3, output_padding=1, padding=1, stride=2),  # [64, 20, 18]
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.ConvTranspose2d(64, 32, kernel_size=3, output_padding=1, padding=1, stride=2),  # [32, 40, 36]
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.ConvTranspose2d(32, 32, kernel_size=3, output_padding=1, padding=1, stride=2),  # [32, 80, 72]
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=3, padding=1),  # [32, 80, 72]
    nn.ReLU(),
    nn.ConvTranspose2d(32, 3, kernel_size=3, output_padding=1, padding=1, stride=2),
    nn.Sigmoid()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
).to(device)

if exists(settings.MODEL_PATH + "encoder.m"):
    encoder.load_state_dict(torch.load(settings.MODEL_PATH + "encoder.m", map_location=device))

if exists(settings.MODEL_PATH + "decoder.m"):
    decoder.load_state_dict(torch.load(settings.MODEL_PATH + "decoder.m", map_location=device))

if exists(settings.MODEL_PATH + "core_net.m"):
    core_net.load_state_dict(torch.load(settings.MODEL_PATH + "core_net.m", map_location=device))


def save_encoder():
    torch.save(encoder.state_dict(), settings.MODEL_PATH + "encoder.m")


def save_decoder():
    torch.save(decoder.state_dict(), settings.MODEL_PATH + "decoder.m")


def save_core():
    torch.save(core_net.state_dict(), settings.MODEL_PATH + "core_net.m")

import torch
import torch.nn as nn
import numpy as np
import math
from torch.optim import Adam
from tqdm import tqdm

NUM_ACTIONS = 2
IMAGE_SIZE = 1
WORLD_SIZE = 3
HIDDEN_SIZE = 10
REWARD_SIZE = 1

POSSIBLE_ACTIONS = ["RIGHT", "LEFT", "NONE"]

loss_fn = nn.MSELoss()

core_net = nn.Sequential(nn.Linear(NUM_ACTIONS + WORLD_SIZE + IMAGE_SIZE, HIDDEN_SIZE),
                         nn.ReLU(),
                         nn.Linear(HIDDEN_SIZE, WORLD_SIZE + REWARD_SIZE + IMAGE_SIZE),
                         nn.Tanh())
core_net_optimizer = Adam(core_net.parameters(), lr=0.0001, weight_decay=0.0001)

imagine_net = nn.Sequential(nn.Linear(WORLD_SIZE, HIDDEN_SIZE),
                            nn.ReLU(),
                            nn.Linear(HIDDEN_SIZE, IMAGE_SIZE),
                            nn.Tanh())

ponder_net = nn.Sequential(nn.Linear(IMAGE_SIZE * 2, HIDDEN_SIZE),
                           nn.ReLU(),
                           nn.Linear(HIDDEN_SIZE, NUM_ACTIONS),
                           nn.Tanh())

action = np.random.choice(POSSIBLE_ACTIONS, 50)
a2 = []
data = [0]
sequence = [[0, 0, 0]]
for a in action:
    if a == "RIGHT":
        a2.append([0, 1])
        data.append(data[-1] + 1)
    elif a == "LEFT":
        a2.append([1, 0])
        data.append(data[-1] - 1)
    else:
        data.append(data[-1])
        a2.append([0, 0])

for i in range(len(a2)):
    sequence.append([a2[i][0], a2[i][1], data[i] / 10])

sequence = np.array(sequence)
sequence = torch.from_numpy(sequence)


def generate_images(input_sequence):
    world = torch.from_numpy(np.zeros(WORLD_SIZE))

    generated_images = []
    for step in input_sequence:
        core_input = torch.cat((world, step), dim=0).float()
        world, image, _ = core_net(core_input).split([WORLD_SIZE, IMAGE_SIZE, REWARD_SIZE])
        generated_images.append(image.detach())
    return generated_images


def generate_rewards(generated_images, sequence):
    facit = []

    for i in range(len(generated_images) - 1):
        _, facit_image = sequence[i + 1].split([NUM_ACTIONS, IMAGE_SIZE])
        reward = abs(facit_image - generated_images[i])
        facit.append([facit_image, reward])
    return facit


def train(input_sequence, facit):
    for i in tqdm(range(5000)):
        world = torch.from_numpy(np.zeros(WORLD_SIZE))
        core_net_optimizer.zero_grad()
        loss = None
        for step in zip(input_sequence, facit):
            step_facit = torch.cat(step[1]).float()
            core_input = torch.cat((world, step[0]), dim=0).float()
            world, image, reward = core_net(core_input).split([WORLD_SIZE, IMAGE_SIZE, REWARD_SIZE])
            actual = torch.cat((image, reward)).float()
            if loss:
                loss += loss_fn(actual, step_facit)
            else:
                loss = loss_fn(actual, step_facit)

        loss.backward()
        core_net_optimizer.step()


all_images = generate_images(sequence)

target = generate_rewards(all_images, sequence)

train(sequence, target)

world = torch.from_numpy(np.zeros(WORLD_SIZE))

step = np.array([0, 1, 0])
step = torch.from_numpy(step)

core_input = torch.cat((world, step), dim=0).float()
world, image, reward = core_net(core_input).split([WORLD_SIZE, IMAGE_SIZE, REWARD_SIZE])
print((world, image, reward))

step = np.array([0, 1, 0.1])
step = torch.from_numpy(step)

core_input = torch.cat((world, step), dim=0).float()
world, image, reward = core_net(core_input).split([WORLD_SIZE, IMAGE_SIZE, REWARD_SIZE])
print((world, image, reward))

step = np.array([0, 1, 0.2])
step = torch.from_numpy(step)

core_input = torch.cat((world, step), dim=0).float()
world, image, reward = core_net(core_input).split([WORLD_SIZE, IMAGE_SIZE, REWARD_SIZE])
print((world, image, reward))

step = np.array([0, 0, 0.3])
step = torch.from_numpy(step)

core_input = torch.cat((world, step), dim=0).float()
world, image, reward = core_net(core_input).split([WORLD_SIZE, IMAGE_SIZE, REWARD_SIZE])
print((world, image, reward))
step = np.array([0, 0, 0.3])
step = torch.from_numpy(step)

core_input = torch.cat((world, step), dim=0).float()
world, image, reward = core_net(core_input).split([WORLD_SIZE, IMAGE_SIZE, REWARD_SIZE])
print((world, image, reward))

step = np.array([0, 0, 0.3])
step = torch.from_numpy(step)

core_input = torch.cat((world, step), dim=0).float()
world, image, reward = core_net(core_input).split([WORLD_SIZE, IMAGE_SIZE, REWARD_SIZE])
print((world, image, reward))

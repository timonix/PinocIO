import os
import pickle

import numpy as np
import retro
import torch
import random

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

image_channels = 3
latent_dim = 10
world_state_size = 200
action_space = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Networks
encoder = Encoder(image_channels, latent_dim).to(device)
decoder = Decoder(image_channels, world_state_size + action_space).to(device)
action_net = Action(world_state_size, world_state_size, action_space).to(device)
reward_net = RewNet(world_state_size + action_space, 1).to(device)

criterion = nn.MSELoss()

if exists("gym_encoder.m"):
    encoder.load_state_dict(torch.load("gym_encoder.m"))
    encoder.eval()

if exists("gym_decoder.m"):
    decoder.load_state_dict(torch.load("gym_decoder.m"))
    decoder.eval()

cunk_size = 64
cc = cunker(cunk_size)


def cunk(img):
    image_as_array = img.astype(np.uint8)
    pil_image = Image.fromarray(image_as_array)
    return cc.cunk(pil_image)


def save(data, frame_num, sequence_id, name, path):
    dir = path+sequence_id
    if not os.path.exists(dir):
        os.mkdir(dir)
    full_path = path+sequence_id+"/"+name+"_"+sequence_id+"_"+str(frame_num).rjust(5,"0")+".dat"
    with mgzip.open(full_path, 'wb') as f:
        pickle.dump(data, f)


def main():
    sequence_id = random.choice(list(english_words_set))

    # placeholder data
    world_state = torch.FloatTensor(np.zeros(200)).to(device)
    image = np.zeros((256, 320, 3))

    env = retro.make(game='Airstriker-Genesis')
    obs = env.reset()

    frame_num = 0
    next_action = [] * 12
    while True:
        frame_num = frame_num + 1
        obs, rew, done, info = env.step(next_action)
        # env.render()
        image[:obs.shape[0], :obs.shape[1], :obs.shape[2]] = obs
        cunked_frame = cunk(image)
        ll = []
        for c in cunked_frame:
            ll.append(np.array(c))
        main_input = np.array(ll)
        main_input = main_input / 255
        main_input = np.swapaxes(main_input, 1, 3)
        main_input = torch.FloatTensor(main_input).to(device)

        latent_space = encoder(main_input).reshape((200))
        act, world_state = action_net.forward(latent_space, hidden_state=world_state)

        image_as_array = image.astype(np.uint8)
        pil_image = Image.fromarray(image_as_array)
        decoder_input = torch.hstack((world_state, act))
        decoder_input = torch.reshape(decoder_input, (1, 212))

        imagination = decoder(decoder_input)
        facit_image = pil_image.resize((256, 256))
        facit_image = np.array(facit_image)
        facit_image = facit_image / 255
        facit_image = np.swapaxes(facit_image, 0, 2)
        facit_image = torch.FloatTensor(facit_image).to(device)

        reward = facit_image-imagination
        reward *= reward

        save(main_input, frame_num, sequence_id, "main_input", "../data/")
        save(reward, frame_num, sequence_id, "reward", "../data/")
        save(latent_space, frame_num, sequence_id, "latent_space", "../data/")
        save(world_state, frame_num, sequence_id, "world_state", "../data/")
        save(act, frame_num, sequence_id, "act", "../data/")
        save(facit_image, frame_num, sequence_id, "image", "../data/")

        estimated_reward = reward_net.forward(torch.hstack((world_state, act)))

        next_action = np.round(act.cpu().detach().numpy())

        if frame_num % 1 == 0:
            print(frame_num)

        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()

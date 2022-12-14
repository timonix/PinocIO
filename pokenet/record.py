import os

SHOW_IMAGES = False

import torch
from pyboy import PyBoy
from pyboy import WindowEvent
import random
from PIL import Image, ImageOps
from pokenet import ACTIONS, WORLD_SIZE, IMAGE_SIZE, REWARD_SIZE
from torchvision import transforms
import pokenet

import settings

from tqdm import tqdm
from pokenet import encoder, core_net, decoder

encoder = encoder.to("cpu")
decoder = decoder.to("cpu")
core_net = core_net.to("cpu")

from english_words import english_words_lower_alpha_set

wordList = list(english_words_lower_alpha_set)

import pickle

convert_tensor = transforms.ToTensor()
getimage = transforms.ToPILImage()

action_matrix = torch.eye(len(ACTIONS))

def clean(obj):
    if obj != None:
        del(obj)

for i in range(5):
    with PyBoy('Pmon', disable_renderer=False) as pyboy:
        pyboy.set_emulation_speed(0)
        pyboy.send_input(WindowEvent.STATE_LOAD)

        world = torch.zeros(WORLD_SIZE)

        session_name = random.choice(wordList)

        predicted_latent = None

        reward_all = [0.0]
        actions_all = []
        session_dir = settings.DATA_PATH + session_name
        image_dir = session_dir + "/images/"
        os.mkdir(session_dir)
        os.mkdir(image_dir)
        image = None
        image_file = None
        tensor_image = None
        latent = None
        win = None
        core_input = None
        predicted_reward = None

        for i in tqdm(range(100000)):

            clean(image)
            image = pyboy.screen_image()

            clean(image_file)
            image_file = image_dir + session_name + "_" + str(i) + ".png"
            with open(image_file, 'wb') as image_file:
                image.save(image_file)
                image_file.close()

            clean(tensor_image)
            tensor_image = convert_tensor(image)
            tensor_image.detach()
            image.close()
            tensor_image /= 255.0
            tensor_image = torch.reshape(tensor_image, (1, tensor_image.shape[0], tensor_image.shape[1], tensor_image.shape[2]))

            clean(latent)
            latent = encoder(tensor_image)
            latent.detach()
            if i % 1000 == 0 and SHOW_IMAGES:
                getimage(decoder(latent)[0] * 255.0).show()

            latent = latent[0]
            clean(win)
            win = torch.stack([world] * len(ACTIONS), dim=0)
            latent_space = torch.stack([latent] * len(ACTIONS), dim=0)

            clean(core_input)
            core_input = torch.cat((win, latent_space, action_matrix), dim=1).float()
            core_input.detach()
            core = core_net(core_input)

            if predicted_latent is not None:
                predicted_latent.detach()
                reward = torch.norm(predicted_latent - latent_space)
                reward_all.append(reward.item())
                del(reward)

            clean(world)
            world, predicted_latent, predicted_reward = core.split([WORLD_SIZE, IMAGE_SIZE, REWARD_SIZE], dim=1)

            action_index = (random.choices(range(len(ACTIONS)), weights=predicted_reward, k=1))
            clean(predicted_reward)

            action_index = action_index[0]
            actions_all.append(action_index)
            world = world[action_index].detach()

            pyboy.tick()
            pyboy.tick()
            pyboy.tick()
            pyboy.tick()
            pyboy.tick()
            pyboy.tick()
            pyboy.tick()
            act = ACTIONS[action_index]

            pyboy.send_input(act[0])
            pyboy.tick()
            pyboy.tick()
            pyboy.send_input(act[1])
            pyboy.tick()
            pyboy.tick()

        pyboy.send_input(WindowEvent.STATE_SAVE)
        pyboy.tick()
        pyboy.tick()
        pyboy.tick()


        reward_file = session_dir + "/" + session_name + ".prew"
        file = open(reward_file, 'wb')
        pickle.dump(reward_all, file)
        file.close()

        action_file = session_dir + "/" + session_name + ".pact"
        file = open(action_file, 'wb')
        pickle.dump(actions_all, file)
        file.close()

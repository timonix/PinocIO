import numpy as np
import retro
import torch

from PIL import Image
from cunker import cunker, decunker

from os.path import exists

from template_AI.network_collection import Encoder64 as Encoder
from template_AI.network_collection import Decoder64 as Decoder
from template_AI.network_collection import RNN as Action
from template_AI.network_collection import FC as RewNet


from tqdm import tqdm
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn

image_channels = 3
encoder_base_size = 32
decoder_base_size = 32
latent_dim = 10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(image_channels, latent_dim).to(device)
decoder = Decoder(image_channels, latent_dim).to(device)

criterion = nn.MSELoss()

if exists("encoder.m"):
    encoder.load_state_dict(torch.load("encoder.m"))
    decoder.load_state_dict(torch.load("decoder.m"))

    encoder.eval()
    decoder.eval()

def cunk(img):
    random_array = img.astype(np.uint8)
    pil_image = Image.fromarray(random_array)

    image_size = (320, 256)
    cunk_size = 64

    cc = cunker(cunk_size)
    dc = decunker(image_size, cunk_size)
    cc.cunk(pil_image)

    return cc.cunk(pil_image)


def main():
    env = retro.make(game='Airstriker-Genesis')
    obs = env.reset()
    print(env.action_space)
    image = np.zeros((256, 320, 3))

    action_net = Action(200, 200, 12).to(device)
    reward_net = RewNet(200+12, 1).to(device)
    world_state = torch.FloatTensor(np.zeros(200)).to(device)

    frame_num = 0
    next_action = []*12
    while True:
        frame_num = frame_num +1
        obs, rew, done, info = env.step(next_action)
        #env.render()
        image[:obs.shape[0], :obs.shape[1], :obs.shape[2]] = obs
        cunked_frame = cunk(image)
        ll = []
        for c in cunked_frame:
            ll.append(np.array(c))
        x = np.array(ll)
        x = x/255
        x = np.swapaxes(x, 1, 3)
        x = torch.FloatTensor(x).to(device)

        facit = x

        latent_space = encoder(x)
        output = decoder(latent_space)

        act_input = latent_space.reshape((200))
        act, world_state = action_net.forward(act_input, hidden_state=world_state)

        next_action = np.round(act.cpu().detach().numpy())
        rew = criterion(output, facit)
        estimated_reward = reward_net.forward(torch.hstack((world_state, act)))

        rew_facit = rew

        rrr = criterion(estimated_reward, rew)
        next_action = np.round(act.cpu().detach().numpy())
        if frame_num % 1 == 0:
            print(frame_num)


        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()


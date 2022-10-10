import torch
from torch import nn

import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2output = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state))
        hidden = self.in2hidden(combined)
        output = torch.sigmoid(self.in2output(combined))
        return output, hidden

    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))


class Encoder64(nn.Module):
    def __init__(self,
                 num_input_channels: int,
                 latent_dim: int,
                 layer_params=None,  # small to large
                 act_fn: object = torch.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        if layer_params is None:
            layer_params = [32, 32, 32, 32, 32, 64, 64]
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, layer_params[0], kernel_size=3, padding=1, stride=2),  # 64x64 => 32x32
            act_fn(),
            nn.Conv2d(layer_params[0], layer_params[1], kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(layer_params[1], layer_params[2], kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(layer_params[2], layer_params[3], kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(layer_params[3], layer_params[4], kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(layer_params[4], layer_params[5], kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(layer_params[5], layer_params[6], kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(16 * layer_params[6], latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decoder64(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 latent_dim: int,
                 layer_params=None,
                 act_fn: object = torch.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        if layer_params is None:
            layer_params = [64, 64, 32, 32, 32, 32, 32, 32]

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 16 * layer_params[0]),
            act_fn()
        )
        self.net = nn.Sequential(  # large to small
            nn.ConvTranspose2d(layer_params[0], layer_params[1], kernel_size=3, output_padding=1, padding=1, stride=2),
            # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(layer_params[1], layer_params[2], kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(layer_params[2], layer_params[3], kernel_size=3, output_padding=1, padding=1, stride=2),
            # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(layer_params[3], layer_params[4], kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(layer_params[4], layer_params[5], kernel_size=3, output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            act_fn(),
            nn.Conv2d(layer_params[5], layer_params[6], kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(layer_params[6], num_input_channels, kernel_size=3, output_padding=1, padding=1,
                               stride=2),

            nn.Sigmoid()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


class Decoder256(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 latent_dim: int,
                 params=None,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        if params is None:
            params = [64, 64, 32, 32, 32, 32, 32, 32, 32, 32]

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 16 * params[0]),
            act_fn()
        )
        self.net = nn.Sequential(  # large to small
            nn.ConvTranspose2d(params[0], params[1], kernel_size=3, output_padding=1, padding=1, stride=2),
            # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(params[1], params[2], kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(params[2], params[3], kernel_size=3, output_padding=1, padding=1, stride=2),
            # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(params[3], params[4], kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(params[4], params[5], kernel_size=3, output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            act_fn(),
            nn.Conv2d(params[5], params[6], kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(params[6], num_input_channels, kernel_size=3, output_padding=1, padding=1,stride=2),
            # 32x32 => 64x64
            act_fn(),
            nn.Conv2d(params[6], params[7], kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(params[7], params[8], kernel_size=3, output_padding=1, padding=1,stride=2),
            # 64x64 => 128x128
            act_fn(),
            nn.Conv2d(params[8], params[9], kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(params[9], num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 128x128 => 256x256

            nn.Sigmoid()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


class Encoder_B(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decoder_B(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            nn.Sigmoid()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


class Encoder(nn.Module):
    def __init__(self, width, height, num_filters_layer_one, num_filters_layer_two, latent_size):
        super(Encoder, self).__init__()

        self.c1 = nn.Conv2d(3, num_filters_layer_one, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.c2 = nn.Conv2d(num_filters_layer_one, num_filters_layer_two, kernel_size=(3, 3), stride=(2, 2),
                            padding=(1, 1))
        self.c_lin = nn.Linear(int(width * height * num_filters_layer_two / 4), latent_size)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c_lin(x))
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size * 2, input_size * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size * 4, output_size),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.encoder(x)


class FC(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FC, self).__init__()
        # Our first linear layer take input_size, in this case 784 nodes to 50
        # and our second linear layer takes 50 to the num_classes we have, in
        # this case 10.
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        """
        x here is the mnist images and we run it through fc1, fc2 that we created above.
        we also add a ReLU activation function in between and for that (since it has no parameters)
        I recommend using nn.functional (F)
        """

        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

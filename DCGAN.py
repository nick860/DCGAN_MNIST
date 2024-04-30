import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, featuers_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x channels_img x 64 x 64 where N is batch size
            nn.Conv2d(
                channels_img, featuers_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(featuers_d, featuers_d * 2, 4, 2, 1), # shape of (N, featuers_d * 2, 32, 32)
            self._block(featuers_d * 2, featuers_d * 4, 4, 2, 1), # shape of (N, featuers_d * 4, 16, 16)
            self._block(featuers_d * 4, featuers_d * 8, 4, 2, 1), # shape of (N, featuers_d * 8, 8, 8)
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(featuers_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # use batch norm to stabilize training
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.disc(x)
    

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, featuers_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim, featuers_g * 16, 4, 1, 0),  # in shape (N, featuers_g * 16, 4, 4)
            self._block(featuers_g * 16, featuers_g * 8, 4, 2, 1),  # in shape (N, featuers_g * 8, 8, 8)
            self._block(featuers_g * 8, featuers_g * 4, 4, 2, 1),  #  in shape (N, featuers_g * 4, 16, 16)
            self._block(featuers_g * 4, featuers_g * 2, 4, 2, 1),  #  in shape (N, featuers_g * 2, 32, 32)
            nn.ConvTranspose2d(featuers_g * 2, channels_img, kernel_size=4, stride=2, padding=1), # out shape (N, channels_img, 64, 64)
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)
    

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02) # mean=0, std=0.02 different from normal distribution

    
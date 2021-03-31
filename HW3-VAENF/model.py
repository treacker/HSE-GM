import torch
from torch import nn
from torch.nn import functional as F

import numpy as np


class Block(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride=1, bias=False, upsample=False, bn=True, activ=True):
        super().__init__()

        layers = []

        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        layers.append(nn.Conv2d(in_features, out_features, kernel, stride=stride, padding=(kernel-1)//2, bias=bias))

        if bn:
            layers.append(nn.BatchNorm2d(out_features))
        if activ:
            layers.append(nn.LeakyReLU(0.2))

        self.block = nn.Sequential(*layers)
    def forward(self, x):

        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, hidden_dim, input_dim=3):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            Block(input_dim, 32, 3, stride=2),
            Block(32, 64, 3, stride=2),
            Block(64, 128, 3, stride=2),
            Block(128, 256, 3, stride=2, bn=False, activ=False),

            nn.Flatten()
            
        )

        self.mu = nn.Linear(256 * 16, hidden_dim)
        self.sigma = nn.Linear(256 * 16, hidden_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu, sigma = self.mu(x), self.sigma(x)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim=3):
        super(Decoder, self).__init__()

        self.decoder_input = nn.Linear(hidden_dim, 256 * 16)

        self.decoder = nn.Sequential(
            Block(256, 128, 3, upsample=True),
            Block(128, 64, 3, upsample=True),
            Block(64, 32, 3, upsample=True),
            Block(32, output_dim, 3, upsample=True, bn=False, activ=False),
            nn.Tanh()
            
        )

    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 4, 4)
        
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self, device, input_dim=3, hidden_dim=128):
        super(VAE, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(hidden_dim=hidden_dim, input_dim=input_dim).to(device)
        self.decoder = Decoder(hidden_dim=hidden_dim, output_dim=input_dim).to(device)


    def train(self):
        self.encoder.train()
        self.decoder.train()
        
    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        
        z = self.reparameterize(mu, logvar)
        
        output = self.decoder(z)
        
        return output, mu, logvar

    def sample(self, n_samples):
        z = torch.randn(n_samples,
                        self.hidden_dim)

        z = z.to(self.device)

        samples = self.decoder(z)

        return samples


    def generate(self, x):
        out, _, _ = self.forward(x)
        return out


def loss_function(recon_x, x, mu, logvar):
    batch_size = recon_x.shape[0]
    MSE = F.mse_loss(recon_x.view(batch_size,-1), x.view(batch_size, -1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD, MSE, KLD
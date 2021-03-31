import torch
from torch import nn
from torch.nn import functional as F
from utils import compute_gradient_penalty, permute_labels
import numpy as np

from torch.nn.utils import spectral_norm as SpectralNorm


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.residual(x)

class GANLoss(nn.Module):
    def __init__(self, device):

        super(GANLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss().to(device)
    
    def __call__(self, prediction, target):
        # print(target)
        # print(prediction.dtype, target.dtype)
        target = target.type(prediction.dtype)
        
        return self.loss(prediction, target)

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['tanh', nn.Tanh()],
        ['sigmoid', nn.Sigmoid()],
        ['none', nn.Identity()],
        ['leaky', nn.LeakyReLU(0.2, inplace=True)]
    ])[activation]

class UnetBlock(nn.Module):
    """(Conv2d, BN, Activation) x 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, 
                 kernel_size=3, padding=1, stride=1, activation='relu'):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.activation = activation_func(activation)
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            self.activation,
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            self.activation
        )

    def forward(self, x):
        x = self.layers(x)
        return x



class Generator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=5, n_layers=6):
        super().__init__()
        
        layers = []
        layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(n_layers):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim //= 2 

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x, labels):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        labels = labels.view(labels.size(0), labels.size(1), 1, 1)
        labels = labels.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, labels], dim=1)
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, n_layers=4, in_channels=3, features=32):
        super(UNet, self).__init__()
        
        self.out_channels = in_channels
        self.n_layers = n_layers

        self.encoders = nn.ModuleList()

        in_sizes = [in_channels + 3] + [features * (2 ** i) for i in range(n_layers - 1)]
        out_sizes = [features * (2 ** i) for i in range(n_layers)]
        
        for (in_size, out_size) in zip(in_sizes, out_sizes):
            self.encoders.append(
                UnetBlock(in_size, out_size)    
            )

        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)

        self.f_map = UnetBlock(features * (2 ** (n_layers - 1)), features * (2 ** n_layers))

        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        in_sizes = [features * (2 ** i) for i in range(1, n_layers + 1)]
        out_sizes = [features * (2 ** i) for i in range(n_layers)]

        for (in_size, out_size) in zip(in_sizes, out_sizes):
            self.decoders.append(
                UnetBlock(in_size, out_size)
            )
            self.upsamples.append(
                nn.ConvTranspose2d(
                    in_size, out_size, kernel_size=2, stride=2
                )   
            )

        self.out_conv = nn.Sequential(
            nn.Conv2d(features, 3, 1),
            nn.Tanh()
        )

    def forward(self, x, labels):
        labels = labels.view(labels.size(0), labels.size(1), 1, 1)
        labels = labels.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, labels], dim=1)
        
        x_s = []
        for i in range(self.n_layers):
            x = self.encoders[i](x)
            x_s.append(x)
            x = self.pooling(x)
            
        x = self.f_map(x)

        for i in range(self.n_layers - 1, -1, -1):
            x = self.upsamples[i](x)
            x = torch.cat((x, x_s[i]), dim=1)
            x = self.decoders[i](x)

        x = self.out_conv(x)
        return x

        
class Critic(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, n_layers=6):
        super().__init__()
        layers = []
        layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim
        for i in range(1, n_layers):
            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.1))
            curr_dim *= 2

        kernel_size = int(image_size / np.power(2, n_layers))
        
        self.backbone = nn.Sequential(*layers)
        
        self.image_conv = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.class_conv = SpectralNorm(nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False))
        
    def forward(self, x):
        h = self.backbone(x)
        out_src = self.image_conv(h)
        out_cls = self.class_conv(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

        
class StarGAN:
    def __init__(self, device, g_conv_dim=64, g_n_layers=6, image_size=128, d_conv_dim=64, d_n_layers=6, c_dim=5):
        # self.G = Generator(g_conv_dim, c_dim, g_n_layers).to(device)
        self.G = UNet(g_n_layers, c_dim, g_conv_dim).to(device)
        self.D = Critic(image_size, d_conv_dim, c_dim, d_n_layers).to(device)
        
        self.device = device

        self.optimizators = {
            'G': torch.optim.Adam(self.G.parameters(), lr=0.0001),
            'D': torch.optim.Adam(self.D.parameters(), lr=0.0005),   
        }

        self.criterionGAN = GANLoss(device)
        self.criterionL1 = nn.L1Loss().to(device)
        
    def train(self):
        self.G.train()
        self.D.train()
        
    def eval(self):
        self.G.eval()
        self.D.eval()

    def set_requires_grad(self, net, requires_grad):
        for param in net.parameters():
            param.requires_grad = requires_grad
        
    def trainG(self, image, orig_label, new_label):

        # self.set_requires_grad(self.G, True)
        # self.set_requires_grad(self.D, False)

        self.optimizators['G'].zero_grad()

        fake_images = self.G(image, new_label)
        out_fake, class_fake = self.D(fake_images)
        
        g_loss_fake = -torch.mean(out_fake)
        g_loss_cls = self.criterionGAN(class_fake, new_label)
        

        x_reconst = self.G(fake_images, orig_label)
        # g_loss_rec = self.criterionL1(image, x_reconst)
        g_loss_rec = self.criterionL1(image, x_reconst)

        g_loss = g_loss_fake + 10 * g_loss_rec + 2 * g_loss_cls

        self.optimizators['D'].zero_grad()
        

        g_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1)

        self.optimizators['G'].step()
        
        return g_loss

       
    def trainD(self, image, orig_label, new_label):
        # self.set_requires_grad(self.G, False)
        # self.set_requires_grad(self.D, True)

        self.optimizators['D'].zero_grad()

        with torch.no_grad():
            fake_images = self.G(image, new_label)

        out_real, class_real = self.D(image)
        out_fake, class_fake = self.D(fake_images)

        d_loss_cls = self.criterionGAN(class_real, orig_label)

        d_loss_adv = torch.mean(out_fake) - torch.mean(out_real)

        d_loss = d_loss_adv + 2 * d_loss_cls 
        

        d_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1)
        
        self.optimizators['D'].step()

        return d_loss




    def generate(self, image, label):
        return self.G(image, label)





import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride=1, bias=False, upsample=False):
        super().__init__()
        self.upsample = upsample

        self.conv = nn.Conv2d(in_features, out_features, kernel, stride=stride, padding=(kernel-1)//2, bias=bias)
        self.norm = nn.BatchNorm2d(out_features)
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return self.act(self.norm(self.conv(x)))

class Classifier(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            Block(1, 16, 3, stride=1),
            nn.MaxPool2d(2, 2),
            Block(16, 32, 3, stride=1),
            nn.MaxPool2d(2, 2),
            Block(32, 32, 3, stride=1),
            nn.MaxPool2d(2, 2)
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.dense(x)
        return x
    
    def get_activations(self, x):
        return self.encoder(x)
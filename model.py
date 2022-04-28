import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        ######################
        # write your code here
        self.layer1 = nn.Conv2d(3, 64, 9, stride=1, padding=4)
        self.layer2 = nn.Conv2d(64, 32, 1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(32, 3, 5, stride=1, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        ######################
        # write your code here
        x = F.interpolate(x, scale_factor=3, mode='bicubic')
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np
from models.BaseModel import BaseModel
from models.ModelFactory import ModelFactory

class ResBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        """
            :param in_planes: input channels
            :param planes: output channels
            :param stride: The stride of first conv
        """
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
           self.shortcut = nn.Sequential(
               nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(planes)
           )

    def forward(self, x):
        # 1. Go through conv1, bn1, relu
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        # 2. Go through conv2, bn
        output = self.conv2(output)
        output = self.bn2(output)

        # 3. Combine with shortcut output, and go through relu
        output += self.shortcut(x)
        output = self.relu(output)

        return output
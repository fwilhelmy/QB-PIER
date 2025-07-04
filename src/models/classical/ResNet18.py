import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.decomposition import PCA  # Import PCA for channel reduction
from models.BaseModel import BaseModel
from typing import Tuple

class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        """
            :param in_planes: input channels
            :param planes: output channels
            :param stride: The stride of first conv
        """
        super(BasicBlock, self).__init__()
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


class ResNet18(BaseModel):
    def __init__(self, input_size: Tuple[int, int], num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters("input_size", "num_classes")
        in_channels, _ = input_size
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.linear = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, in_planes, planes, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.linear(output)
        return output

    def visualize(self, logdir='./'):
        """Visualize and save a square grid of all kernel weights for the first conv layer in RGB.
        
        Args:
            logdir (str): Directory where the figure will be saved.
        """
        # Always use the first conv layer (self.conv1)
        module = self.conv1
        filters = module.weight.data.clone()  # [out_channels, in_channels, k, k]
        
        # Determine number of filters (output channels) and grid size
        in_ch = filters.shape[1]
        n_filters = filters.shape[0]
        grid_size = int(math.ceil(math.sqrt(n_filters)))

        # Prepare each filter for display
        filters_np = filters.cpu().numpy()
        images = []
        for i in range(n_filters):
            f = filters_np[i]
            # Normalize each filter individually to [0, 1]
            f_max, f_min = f.max(), f.min()
            f = (f - f_min) / (f_max - f_min if f_max - f_min > 0 else 1)
            if in_ch == 3:
                # Transpose to (height, width, channels) for matplotlib.
                img = np.transpose(f, (1, 2, 0))
            else:
                # Average across channels for visualization
                img = f.mean(axis=0)
            images.append(img)
        
        # Create the square grid of subplots.
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
        axs = axs.flatten()
        
        for idx, ax in enumerate(axs):
            if idx < len(images):
                if in_ch == 3:
                    ax.imshow(images[idx])
                else:
                    ax.imshow(images[idx], cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
            else: ax.axis('off')
                
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(f"{logdir}/conv1.png")
        plt.close(fig)
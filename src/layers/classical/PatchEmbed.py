from torch import nn
import math
import matplotlib.pyplot as plt
import math
import torch
import matplotlib.pyplot as plt
from models.BaseModel import BaseModel
from models.ModelFactory import ModelFactory

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, img_size, patch_size, in_chans, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        :param x: image tensor of shape [batch, channels, img_size, img_size]
        :return out: [batch. num_patches, embed_dim]
        """
        _, _, H, W = x.shape
        assert H == self.img_size, f"Input image height ({H}) doesn't match model ({self.img_size})."
        assert W == self.img_size, f"Input image width ({W}) doesn't match model ({self.img_size})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2) # BCHW -> BNC
        return x
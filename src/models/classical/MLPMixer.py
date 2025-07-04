from torch import nn
import torch
import math
import matplotlib.pyplot as plt
import os
import math
import torch
import matplotlib.pyplot as plt
from models.BaseModel import BaseModel
from typing import Tuple

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


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(
            self,
            in_features,
            hidden_features,
            act_layer=nn.GELU,
            drop=0.,
    ):
        super(Mlp, self).__init__()
        out_features = in_features
        hidden_features = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0),
            activation='gelu', drop=0., drop_path=0.):
        super(MixerBlock, self).__init__()
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}[activation]
        tokens_dim, channels_dim = int(mlp_ratio[0] * dim), int(mlp_ratio[1] * dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # norm1 used with mlp_tokens
        self.mlp_tokens = Mlp(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6) # norm2 used with mlp_channels
        self.mlp_channels = Mlp(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # Token mixing
        residual = x
        output = self.norm1(x)
        output = output.transpose(1, 2) # [batch, embed_dim, num_patches]
        output = self.mlp_tokens(output) # Apply MLP on token dimension
        output = output.transpose(1, 2) # [batch, num_patches, embed_dim]
        output = output + residual # Skip-connection

        # Channel mixing
        residual = output
        output = self.norm2(output)
        output = self.mlp_channels(output) # Apply MLP on channel dimension
        output = output + residual # Skip-connection

        return output
    

class MLPMixer(BaseModel):
    def __init__(self, num_classes, input_size: Tuple[int, int], patch_size, embed_dim, num_blocks, 
                 drop_rate=0., activation='gelu', mlp_ratio=(0.5, 4.0), **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters("num_classes", "input_size", "patch_size", "embed_dim",
                                 "num_blocks", "drop_rate", "activation", "mlp_ratio")
        in_chans, img_size = input_size
        self.patchemb = PatchEmbed(img_size=img_size, 
                                   patch_size=patch_size, 
                                   in_chans=in_chans,
                                   embed_dim=embed_dim)
        self.blocks = nn.Sequential(*[
            MixerBlock(
                dim=embed_dim, seq_len=self.patchemb.num_patches, 
                mlp_ratio=mlp_ratio, activation=activation, drop=drop_rate)
            for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, inputs):
        """ MLPMixer forward
        :param images: [batch, in_chans, img_size, img_size]
        """

        # Step 1 : Convert images to a sequence of patch embeddings
        output = self.patchemb(inputs) # [batch, num_patches, embed_dim]
        # Step 2 : Go through the mixer blocks
        output = self.blocks(output) # [batch, num_patches, embed_dim]
        # Step 3 : Layer norm
        output = self.norm(output) # [batch, num_patches, embed_dim]
        # Step 4 : Global average pooling over the patch (token) dimension
        output = output.mean(dim=1) # [batch, embed_dim]
        # Step 5 : Final classification head
        output = self.head(output) # [batch, num_classes]

        return output

    def visualize(self, block_idx, logdir='./'):
        """
        Visualize and save a square grid of the token mixing weights for a given MixerBlock.

        Args:
            block_idx (int): The index of the MixerBlock to visualize.
            logdir (str): Directory where the figure will be saved.
        """
        assert block_idx > 0 or block_idx <= len(self.blocks), f"block_index={block_idx} is out of range for {len(self.blocks)} MixerBlocks."

        # Select the specified mixer block (keep your existing block selection logic)
        block = self.blocks[block_idx]

        # Retrieve the weights from the token mixing linear layer.
        weights = block.mlp_tokens.fc1.weight.data.clone() # [out_features, in_features]
        out_features, in_features = weights.shape

        # Check if in_features is a perfect square (so we can reshape the weight vector into a square image)
        kernel_side = int(math.sqrt(in_features))
        reshape_kernel = (kernel_side * kernel_side == in_features)

        # Prepare kernels: normalize each weight vector and reshape if possible.
        weights_np = weights.cpu().numpy()
        kernels = []
        for i in range(out_features):
            kernel = weights_np[i]
            # Normalize each kernel individually to [0, 1]
            k_min, k_max = kernel.min(), kernel.max()
            # Edge-case Handling (Division by Zero)
            kernel = (kernel - k_min) / (k_max - k_min if k_max - k_min > 0 else 1)
            if reshape_kernel: kernel = kernel.reshape(kernel_side, kernel_side)
            kernels.append(kernel)

        # Create a square grid to display all kernels (out_features = grid_size * grid_size)
        grid_size = int(math.ceil(math.sqrt(out_features)))
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
        axs = axs.flatten()

        for idx, ax in enumerate(axs):
            if idx < out_features:
                ax.imshow(kernels[idx], cmap='bwr')
                ax.set_xticks([])
                ax.set_yticks([])
            else: ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])        
        fig.savefig(f"{logdir}/mixer_block_{block_idx}_token_mixing.png")
        plt.close(fig)
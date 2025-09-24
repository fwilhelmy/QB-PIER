from torch import nn
import math
import matplotlib.pyplot as plt
import torch

from models import BaseModel, ModelFactory
from layers.classical import MixerBlock, PatchEmbed

@ModelFactory.model("mlpmixer")
class MLPMixer(BaseModel):
    def __init__(self, patch_size, embed_dim, num_blocks,
                 drop_rate=0., activation='gelu', mlp_ratio=(0.5, 4.0), **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters("patch_size", "embed_dim",
                                 "num_blocks", "drop_rate", "activation", "mlp_ratio")
        in_chans, img_size, _ = self.input_size
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
        self.head = nn.Linear(embed_dim, self.output_size)
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
        output = self.head(output) # [batch, output_size]

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
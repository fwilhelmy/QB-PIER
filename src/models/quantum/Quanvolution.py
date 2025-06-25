# quanvolution.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

from src.models.quantum.VQC import VQC
from src.models.quantum.ansatzs.RandomStructureAnsatz import RandomStructureAnsatz
from src.models.quantum.ansatzs.BaseAnsatz import BaseAnsatz


class Quanvolution(nn.Module):
    """
    Quantum convolutional layer:
      - extracts patches of size (patch_size x patch_size)
      - runs each patch through a VQC
      - returns a feature map of dimension (batch, n_qubits, H_out, W_out)
    
    Args:
        patch_size: int, side-length of each square patch
        stride:     int, patch stride (default = patch_size → non-overlapping)
        n_layers:   int, depth of the ansatz inside each VQC
        ansatz_cls: BaseAnsatz subclass (default = BasicEntanglingAnsatz)
        embedding:  'angle' or 'amplitude'
        rotation:   'X','Y','Z' for single-qubit gates in the no-entangling variant
        device:     optional PennyLane device; if None, uses default.qubit
    """
    def __init__(
        self,
        patch_size: int,
        stride: int | None = None,
        n_layers: int = 1,
        ansatz_cls: Type[BaseAnsatz] = RandomStructureAnsatz,
        embedding: str = "angle",
        rotation: str = "X",
        device=None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.n_qubits = patch_size * patch_size

        # Build a VQC that exactly matches this patch size
        self.vqc = VQC(
            n_qubits=self.n_qubits,
            n_layers=n_layers,
            ansatz_cls=ansatz_cls,
            embedding=embedding,
            rotation=rotation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch, channels=1, H, W)
        returns: (batch, n_qubits, H_out, W_out)
        """
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Quanvolution only supports single-channel images for now.")
        
        B, _, H_in, W_in = x.shape
        # Extract patches (B, 1, H_out, W_out, p, p)
        patches = x.unfold(2, self.patch_size, self.stride) \
                   .unfold(3, self.patch_size, self.stride)
        H_out, W_out = patches.size(2), patches.size(3)

        # Flatten each patch (B, H_out, W_out, p*p)
        patches = patches.contiguous().view(B, 1, H_out, W_out, -1)
        patches = patches.view(B, H_out * W_out, self.n_qubits)  # (B, n_patches, n_qubits)

        # Run all patches in one batch through the VQC
        batch_patches, n_patches = B * patches.size(1), patches.size(1)
        inputs = patches.view(batch_patches, self.n_qubits) # (batch_patches, n_qubits)
        out = self.vqc(inputs) # (batch_patches, n_qubits)
        out = out.view(B, n_patches, self.n_qubits) # (B, n_patches, n_qubits)

        # Reassemble into feature map (B, n_qubits, H_out, W_out)
        out = out.permute(0, 2, 1) # (B, n_qubits, n_patches)
        out = out.view(B, self.n_qubits, H_out, W_out) # (B, n_qubits, H_out, W_out)
        return out

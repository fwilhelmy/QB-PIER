import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .VQC import VQC
import torch
from torch import Tensor, flatten
from layers import BaseLayer, LayerFactory


log = logging.getLogger(__name__)

class Quanv2d(nn.Module):
    """Quantum 2D convolution layer using a VQC on image patches."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        n_layers: int = 1,
        embedding: str = "angle",
        ansatz: dict | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.input_size = in_channels * kernel_size * kernel_size
        self.n_qubits = self.input_size

        self.vqc = VQC(
            input_size=self.input_size,
            n_layers=n_layers,
            embedding=embedding,
            ansatz=ansatz
        )
        self.readout = nn.Linear(self.n_qubits, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            message = f"Quanv2d expects input shape (batch, {self.in_channels}, H, W)"
            log.error(message)
            raise ValueError(message)

        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        B = x.size(0)
        L = patches.size(-1)
        patches = patches.transpose(1, 2).reshape(B * L, self.n_qubits)

        out = self.vqc(patches)
        out = self.readout(out)
        out = out.view(B, L, self.out_channels).permute(0, 2, 1)

        H_out = int(
            (x.size(2) + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
            / self.stride
            + 1
        )
        W_out = int(
            (x.size(3) + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
            / self.stride
            + 1
        )
        out = out.reshape(B, self.out_channels, H_out, W_out)
        return out

@LayerFactory.layer("quanv")
class Quanvolution(BaseLayer):
    """Hybrid layer using a quantum convolution block."""

    def __init__(
        self,
        input_size: tuple[int, int, int],
        patch_size: int = 2,
        stride: int | None = None,
        n_qdepth: int = 1,
        ansatz: dict | None = None,
        embedding: str = "angle",
        rotation: str = "X",
    ) -> None:
        super().__init__(input_size=input_size)
        if self.input_size is None:
            raise RuntimeError("Input size must be specified for Quanvolution")
        channels, height, width = self.input_size
        stride = stride or patch_size
        ansatz = ansatz or {}
        self.quanv = Quanv2d(
            in_channels=channels,
            out_channels=patch_size * patch_size,
            kernel_size=patch_size,
            stride=stride,
            padding=0,
            dilation=1,
            n_layers=n_qdepth,
            embedding=embedding,
            ansatz=ansatz,
        )
        n_patches = (height - patch_size) // stride + 1
        self.output_size = self.quanv.out_channels * n_patches * n_patches

    def forward(self, x: Tensor) -> Tensor:
        x = self.quanv(x)
        return flatten(x.to(torch.float32), 1)

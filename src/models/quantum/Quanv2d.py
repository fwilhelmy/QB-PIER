import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

from models.quantum.VQC import VQC
from models.quantum.ansatz.RandomStructureAnsatz import RandomStructureAnsatz
from models.quantum.ansatz.BaseAnsatz import BaseAnsatz

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
        ansatz_cls: Type[BaseAnsatz] = RandomStructureAnsatz,
        embedding: str = "angle"
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.input_size = in_channels * kernel_size * kernel_size

        self.vqc = VQC(
            input_size=self.input_size,
            n_layers=n_layers,
            ansatz_cls=ansatz_cls,
            embedding=embedding
        )
        self.readout = nn.Linear(self.n_qubits, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Quanv2d expects input shape (batch, {self.in_channels}, H, W)"
            )

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

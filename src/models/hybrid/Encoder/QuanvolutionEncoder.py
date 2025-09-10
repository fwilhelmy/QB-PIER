from __future__ import annotations

import torch
from torch import Tensor, flatten

from models.quantum import Quanv2d

from .BaseEncoder import BaseEncoder


class QuanvolutionEncoder(BaseEncoder):
    """Encoder using a quantum convolution layer."""

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
        super().__init__(input_size)
        channels, height, width = input_size
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
        self._out_features = self.quanv.out_channels * n_patches * n_patches

    @property
    def output_size(self) -> int:
        return self._out_features

    def forward(self, x: Tensor) -> Tensor:
        x = self.quanv(x)
        return flatten(x.to(torch.float32), 1)

from __future__ import annotations

import torch
from torch import Tensor, flatten

from models.quantum import Quanv2d

from ..HybridLayer import HybridLayer
from ..HybridLayerFactory import HybridLayerFactory


@HybridLayerFactory.encoder("quanv")
class QuanvolutionEncoder(HybridLayer):
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
        super().__init__(input_size=input_size)
        if self.input_size is None:
            raise RuntimeError("Input size must be specified for QuanvolutionEncoder")
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

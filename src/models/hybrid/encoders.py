from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, flatten
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)
from sklearn.decomposition import PCA

from models.quantum import Quanv2d


class ConvolutionEncoder(Module):
    """Classical convolutional feature extractor."""

    def __init__(
        self,
        input_size: tuple[int, int, int],
        conv_channels: tuple[int, int] = (16, 32),
    ) -> None:
        super().__init__()
        self.input_size = input_size
        c1, c2 = conv_channels

        self.conv1 = Conv2d(
            in_channels=input_size[0],
            out_channels=c1,
            kernel_size=(5, 5),
            padding=2,
        )
        self.batchNorm1 = BatchNorm2d(c1)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=(5, 5),
            padding=2,
        )
        self.batchNorm2 = BatchNorm2d(c2)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self._out_features = self._get_conv_out_size()

    def _get_conv_out_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_size)
            x = self.maxpool1(self.relu1(self.batchNorm1(self.conv1(dummy))))
            x = self.maxpool2(self.relu2(self.batchNorm2(self.conv2(x))))
            return x.view(1, -1).size(1)

    @property
    def output_size(self) -> int:
        return self._out_features

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool1(self.relu1(self.batchNorm1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.batchNorm2(self.conv2(x))))
        return flatten(x, 1)


class QuanvolutionEncoder(Module):
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
        super().__init__()
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


class LinearEncoder(Module):
    """Simple MLP-based encoder operating on flattened inputs."""

    def __init__(
        self,
        input_size: tuple[int, int, int],
        hidden_sizes: int | tuple[int, ...] = (128,),
    ) -> None:
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,)
        layers: list[Module] = []
        prev = int(np.prod(input_size))
        for h in hidden_sizes:
            layers.append(Linear(prev, h))
            layers.append(ReLU())
            prev = h
        self.net = Sequential(*layers)
        self._out_features = prev

    @property
    def output_size(self) -> int:
        return self._out_features

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        return self.net(x)


class PCAEncoder(Module):
    """Encoder applying Principal Component Analysis."""

    def __init__(
        self,
        input_size: tuple[int, int, int],
        n_components: int,
    ) -> None:
        super().__init__()
        self.input_dim = int(np.prod(input_size))
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self._out_features = n_components

    @property
    def output_size(self) -> int:
        return self._out_features

    def fit(self, data: Tensor) -> None:
        flat = data.view(data.size(0), -1).detach().cpu().numpy()
        self.pca.fit(flat)

    def forward(self, x: Tensor) -> Tensor:
        flat = x.view(x.size(0), -1).detach().cpu().numpy()
        transformed = self.pca.transform(flat)
        return torch.tensor(transformed, device=x.device, dtype=x.dtype)


ENCODER_REGISTRY = {
    "conv": ConvolutionEncoder,
    "quanv": QuanvolutionEncoder,
    "linear": LinearEncoder,
    "pca": PCAEncoder,
}

__all__ = [
    "ConvolutionEncoder",
    "QuanvolutionEncoder",
    "LinearEncoder",
    "PCAEncoder",
    "ENCODER_REGISTRY",
]

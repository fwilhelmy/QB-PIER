from __future__ import annotations

import torch
from torch import Tensor, flatten
from torch.nn import BatchNorm2d, Conv2d, MaxPool2d, ReLU

from ..HybridLayer import HybridLayer


class ConvolutionEncoder(HybridLayer):
    """Classical convolutional feature extractor."""

    def __init__(
        self,
        input_size: tuple[int, int, int],
        conv_channels: tuple[int, int] = (16, 32),
    ) -> None:
        super().__init__(input_size=input_size)
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

        self.output_size = self._get_conv_out_size()

    def _get_conv_out_size(self) -> int:
        with torch.no_grad():
            if self.input_size is None:
                raise RuntimeError("Input size must be specified for ConvolutionEncoder")
            dummy = torch.zeros(1, *self.input_size)
            x = self.maxpool1(self.relu1(self.batchNorm1(self.conv1(dummy))))
            x = self.maxpool2(self.relu2(self.batchNorm2(self.conv2(x))))
            return x.view(1, -1).size(1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool1(self.relu1(self.batchNorm1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.batchNorm2(self.conv2(x))))
        return flatten(x, 1)

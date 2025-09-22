from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential

from ..HybridLayer import HybridLayer
from ..HybridLayerFactory import HybridLayerFactory


@HybridLayerFactory.encoder("linear")
class LinearEncoder(HybridLayer):
    """Simple MLP-based encoder operating on flattened inputs."""

    def __init__(
        self,
        input_size: tuple[int, int, int],
        hidden_sizes: int | tuple[int, ...] = (128,),
    ) -> None:
        super().__init__(input_size=input_size)
        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,)
        layers: list[Module] = []
        prev = int(np.prod(input_size))
        for h in hidden_sizes:
            layers.append(Linear(prev, h))
            layers.append(ReLU())
            prev = h
        self.net = Sequential(*layers)
        self.output_size = prev

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        return self.net(x)

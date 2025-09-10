from __future__ import annotations

from torch import Tensor
from torch.nn import Linear, LogSoftmax, Module, ReLU, Sequential

from .BaseClassifier import BaseClassifier


class MLPClassifier(BaseClassifier):
    """Purely classical multilayer perceptron classifier."""

    def __init__(
        self,
        in_features: int,
        output_size: int,
        hidden_sizes: int | tuple[int, ...] = (128,),
    ) -> None:
        super().__init__(in_features, output_size)
        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,)
        layers: list[Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(Linear(prev, h))
            layers.append(ReLU())
            prev = h
        layers.append(Linear(prev, output_size))
        layers.append(LogSoftmax(dim=1))
        self.net = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

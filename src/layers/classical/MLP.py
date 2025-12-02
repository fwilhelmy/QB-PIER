from __future__ import annotations

from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential

from layers import BaseLayer, LayerFactory

@LayerFactory.layer("mlp")
class MLP(BaseLayer):
    """Purely classical multilayer perceptron head."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: int | tuple[int, ...] = (128,),
    ) -> None:
        super().__init__(input_size=input_size, output_size=output_size)
        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,)
        layers: list[Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(Linear(prev, h))
            layers.append(ReLU())
            prev = h
        layers.append(Linear(prev, output_size))
        self.net = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

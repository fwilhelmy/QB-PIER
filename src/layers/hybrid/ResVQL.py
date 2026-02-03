from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, ReLU

from ..BaseLayer import BaseLayer
from ..LayerFactory import LayerFactory
from ..quantum.VQC import VQC


@LayerFactory.layer("resvql")
class ResVQL(BaseLayer):
    """Hybrid classifier with a learnable residual around a quantum block."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        in_qlayer: int,
        n_qdepth: int = 3,
        ansatz: dict | None = None,
        embedding: str = "amplitude",
        residual_init: float = 0.5,
    ) -> None:
        super().__init__(input_size=input_size, output_size=output_size)
        self.fc1 = Linear(input_size, in_qlayer)
        self.bn = BatchNorm1d(in_qlayer)
        self.relu = ReLU()
        self.vqc = VQC(
            input_size=in_qlayer,
            n_layers=n_qdepth,
            embedding=embedding,
            ansatz=ansatz,
        )
        self.residual_proj = Linear(input_size, output_size)
        self.residual_scale = torch.nn.Parameter(
            torch.tensor(residual_init, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        main = self.fc1(x)
        main = self.bn(main)
        main = self.relu(main)
        main = self.vqc(main)
        residual = self.residual_proj(x)
        output = main + self.residual_scale * residual
        return output


__all__ = ["ResVQL"]

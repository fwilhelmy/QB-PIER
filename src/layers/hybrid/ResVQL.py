from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, LogSoftmax, ReLU

from layers import BaseLayer, LayerFactory
from layers.quantum import VQC


@LayerFactory.classifier("resqblock")
class ResVQL(BaseLayer):
    """Hybrid classifier with a learnable residual around a quantum block."""

    def __init__(
        self,
        in_features: int,
        output_size: int,
        in_qlayer: int,
        n_qdepth: int = 3,
        ansatz: dict | None = None,
        embedding: str = "amplitude",
        residual_init: float = 0.5,
    ) -> None:
        super().__init__(in_features=in_features, output_size=output_size)
        self.fc1 = Linear(in_features, in_qlayer)
        self.bn = BatchNorm1d(in_qlayer)
        self.relu = ReLU()
        self.vqc = VQC(
            input_size=in_qlayer,
            n_layers=n_qdepth,
            embedding=embedding,
            ansatz=ansatz,
        )
        self.fc2 = Linear(self.vqc.output_size, output_size)
        self.residual_proj = Linear(in_features, output_size)
        self.residual_scale = torch.nn.Parameter(
            torch.tensor(residual_init, dtype=torch.float32)
        )
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        main = self.fc1(x)
        main = self.bn(main)
        main = self.relu(main)
        main = self.vqc(main)
        main = self.fc2(main)

        residual = self.residual_proj(x)
        output = main + self.residual_scale * residual
        return self.logSoftmax(output)

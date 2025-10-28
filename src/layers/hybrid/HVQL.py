from __future__ import annotations

from torch import Tensor
from torch.nn import BatchNorm1d, Linear, LogSoftmax, ReLU

from layers import BaseLayer, LayerFactory
from layers.quantum import VQC


@LayerFactory.classifier("hvql")
class HVQL(BaseLayer):
    """Variational Quantum Layer used as a classifier module."""

    def __init__(
        self,
        in_features: int,
        output_size: int,
        in_qlayer: int,
        n_qdepth: int = 3,
        ansatz: dict | None = None,
        embedding: str = "amplitude",
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
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        output = self.fc1(x)
        output = self.bn(output)
        output = self.relu(output)
        output = self.vqc(output)
        output = self.fc2(output)
        return self.logSoftmax(output)

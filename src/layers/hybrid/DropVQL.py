from __future__ import annotations

from torch import Tensor
from torch.nn import BatchNorm1d, Dropout, Linear, LogSoftmax, ReLU

from ..BaseLayer import BaseLayer
from ..LayerFactory import LayerFactory
from ..quantum.VQL import VQC


@LayerFactory.classifier("dropvql")
class DropVQL(BaseLayer):
    """Hybrid classifier that injects dropout before the quantum block."""

    def __init__(
        self,
        in_features: int,
        output_size: int,
        in_qlayer: int,
        n_qdepth: int = 3,
        ansatz: dict | None = None,
        embedding: str = "amplitude",
        dropout: float = 0.1,
    ) -> None:
        super().__init__(in_features=in_features, output_size=output_size)
        self.fc1 = Linear(in_features, in_qlayer)
        self.bn = BatchNorm1d(in_qlayer)
        self.relu = ReLU()
        self.dropout = Dropout(dropout)
        self.vqc = VQC(
            input_size=in_qlayer,
            n_layers=n_qdepth,
            embedding=embedding,
            ansatz=ansatz,
        )
        self.fc_out = Linear(self.vqc.output_size, output_size)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        output = self.fc1(x)
        output = self.bn(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.vqc(output)
        output = self.fc_out(output)
        return self.logSoftmax(output)

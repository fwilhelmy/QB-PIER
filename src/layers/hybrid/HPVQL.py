from __future__ import annotations

from regex import P
from torch import Tensor
from torch.nn import Linear, LogSoftmax, ReLU

from layers import BaseLayer, LayerFactory
from layers.quantum import ParallelVQC

@LayerFactory.classifier("hpvql")
class HPVQL(BaseLayer):
    """Classifier combining a linear layer, parallel VQC, and final linear head."""

    def __init__(
        self,
        in_features: int,
        output_size: int,
        in_qlayer: int,
        qblock_size: int,
        n_qdepth: int = 3,
        ansatz: dict | None = None,
        embedding: str = "angle",
    ) -> None:
        super().__init__(in_features=in_features, output_size=output_size)
        self.fc1 = Linear(in_features=in_features, out_features=in_qlayer)
        self.relu = ReLU()
        self.vqc = ParallelVQC(
            input_size=in_qlayer,
            block_size=qblock_size,
            n_layers=n_qdepth,
            ansatz=ansatz,
            embedding=embedding,
        )
        self.fc2 = Linear(self.vqc.output_size, output_size)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        output = self.fc1(x)
        output = self.relu(output)
        output = self.vqc(output)
        output = self.fc2(output)
        return self.logSoftmax(output)

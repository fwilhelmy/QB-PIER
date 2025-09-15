from __future__ import annotations

from torch import Tensor
from torch.nn import Linear, LogSoftmax, ReLU

from models.quantum import ParallelVQC

from ..HybridLayer import HybridLayer


class ParallelVQCClassifier(HybridLayer):
    """Classifier combining a linear layer, ParallelVQC, and final linear head."""

    def __init__(
        self,
        in_features: int,
        output_size: int,
        in_qlayer: int,
        qblock_size: int,
        n_qdepth: int = 3,
        ansatz: dict | None = None,
    ) -> None:
        super().__init__(in_features=in_features, output_size=output_size)
        self.fc1 = Linear(in_features=in_features, out_features=in_qlayer)
        self.relu = ReLU()
        self.qlayer = ParallelVQC(
            input_size=in_qlayer,
            block_size=qblock_size,
            n_layers=n_qdepth,
            ansatz=ansatz,
        )
        self.fc2 = Linear(self.qlayer.output_size, output_size)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.qlayer(x)
        x = self.fc2(x)
        return self.logSoftmax(x)

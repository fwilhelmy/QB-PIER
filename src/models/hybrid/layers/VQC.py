from __future__ import annotations

from torch import Tensor
from torch.nn import Linear, LogSoftmax, ReLU

from models.quantum import VQC as QuantumVQC

from ..HybridLayer import HybridLayer
from ..HybridLayerFactory import HybridLayerFactory


@HybridLayerFactory.classifier("vqc")
class VQC(HybridLayer):
    """Classifier using a single variational quantum circuit."""

    def __init__(
        self,
        in_features: int,
        output_size: int,
        in_qlayer: int,
        n_qdepth: int = 3,
        ansatz: dict | None = None,
        embedding: str = "angle",
    ) -> None:
        super().__init__(in_features=in_features, output_size=output_size)
        self.fc1 = Linear(in_features, in_qlayer)
        self.relu = ReLU()
        self.qlayer = QuantumVQC(
            input_size=in_qlayer,
            n_layers=n_qdepth,
            embedding=embedding,
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

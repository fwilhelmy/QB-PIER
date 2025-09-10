from __future__ import annotations

from torch import Tensor
from torch.nn import Linear, LogSoftmax, Module, ReLU, Sequential

from models.quantum import ParallelVQC, VQC


class ParallelVQCClassifier(Module):
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
        super().__init__()
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


class VQCClassifier(Module):
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
        super().__init__()
        self.fc1 = Linear(in_features, in_qlayer)
        self.relu = ReLU()
        self.qlayer = VQC(
            input_size=in_qlayer,
            n_layers=n_qdepth,
            embedding=embedding,
            ansatz=ansatz,
        )
        self.fc2 = Linear(in_qlayer, output_size)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.qlayer(x)
        x = self.fc2(x)
        return self.logSoftmax(x)


class MLPClassifier(Module):
    """Purely classical multilayer perceptron classifier."""

    def __init__(
        self,
        in_features: int,
        output_size: int,
        hidden_sizes: int | tuple[int, ...] = (128,),
    ) -> None:
        super().__init__()
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


CLASSIFIER_REGISTRY = {
    "parallel_vqc": ParallelVQCClassifier,
    "vqc": VQCClassifier,
    "mlp": MLPClassifier,
}

__all__ = [
    "ParallelVQCClassifier",
    "VQCClassifier",
    "MLPClassifier",
    "CLASSIFIER_REGISTRY",
]

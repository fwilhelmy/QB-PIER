from __future__ import annotations

from torch import Tensor
from torch.nn import BatchNorm1d, Linear, LogSoftmax, Module, ReLU

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
        **qlayer_kwargs,
    ) -> None:
        super().__init__(in_features=in_features, output_size=output_size)
        self.fc1 = Linear(in_features, in_qlayer)
        self.bn = BatchNorm1d(in_qlayer)
        self.relu = ReLU()
        self.vqc = self._init_qlayer(
            input_size=in_qlayer,
            n_layers=n_qdepth,
            embedding=embedding,
            ansatz=ansatz,
            **qlayer_kwargs,
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

    def _init_qlayer(
        self,
        *,
        input_size: int,
        n_layers: int,
        embedding: str,
        ansatz: dict | None,
        **unused_kwargs,
    ) -> Module:
        if unused_kwargs:
            unexpected = ", ".join(sorted(unused_kwargs))
            raise TypeError(f"Unexpected keyword arguments for HVQL: {unexpected}")
        return VQC(
            input_size=input_size,
            n_layers=n_layers,
            embedding=embedding,
            ansatz=ansatz,
        )

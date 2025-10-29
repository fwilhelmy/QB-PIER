from __future__ import annotations

from torch import Tensor
from torch.nn import BatchNorm1d, Linear, LogSoftmax, Module, ReLU

from layers import BaseLayer, LayerFactory
from layers.quantum import VQC, ParaVQC


@LayerFactory.classifier("hvql")
class HVQL(BaseLayer):
    """Variational Quantum Layer used as a classifier module."""

    def __init__(
        self,
        in_features: int,
        output_size: int,
        in_qlayer: int,
        **qlayer_kwargs,
    ) -> None:
        super().__init__(in_features=in_features, output_size=output_size)
        self.fc1 = Linear(in_features, in_qlayer)
        self.bn = BatchNorm1d(in_qlayer)
        self.relu = ReLU()
        self.qlayer = self._qcls(input_size=in_qlayer, **qlayer_kwargs)
        self.fc2 = Linear(self.qlayer.output_size, output_size)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        output = self.fc1(x)
        output = self.bn(output)
        output = self.relu(output)
        output = self.qlayer(output)
        output = self.fc2(output)
        return self.logSoftmax(output)

    @property
    def _qcls(self) -> type[Module]:
        return VQC

@LayerFactory.classifier("hpvql")
class HPVQL(HVQL):
    """Classifier combining a linear layer, parallel VQC, and final linear head."""

    def __init__(
        self,
        in_features: int,
        output_size: int,
        in_qlayer: int,
        **qlayer_kwargs,
    ) -> None:
        super().__init__(
            in_features=in_features,
            output_size=output_size,
            in_qlayer=in_qlayer,
            **qlayer_kwargs,
        )

    @property
    def _qcls(self) -> type[Module]:
        return ParaVQC

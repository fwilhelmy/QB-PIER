from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Module, ReLU

from layers import BaseLayer, LayerFactory
from layers.quantum import VQC, ParaVQC
from qdevice import QuantumDeviceConfig


@LayerFactory.layer("hvql")
class HVQL(BaseLayer):
    """Variational Quantum Layer used as a classifier module."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        in_qlayer: int,
        quantum_config: Mapping[str, Any] | QuantumDeviceConfig | None = None,
        **qlayer_kwargs,
    ) -> None:
        super().__init__(input_size=input_size, output_size=output_size)
        self.fc1 = Linear(input_size, in_qlayer)
        self.bn = BatchNorm1d(in_qlayer)
        self.relu = ReLU()
        self.qlayer = self._qcls(in_qlayer=in_qlayer, quantum_config=quantum_config, **qlayer_kwargs)
        self.fc2 = Linear(self.qlayer.output_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        output = self.fc1(x)
        output = self.bn(output)
        output = self.relu(output)
        output = self.qlayer(output)
        output = self.fc2(output)
        return output

    def _qcls(self, in_qlayer, **qlayer_kwargs) -> Module:
        qlayer_kwargs['n_layers'] = qlayer_kwargs.pop("n_qdepth", None)
        return VQC(input_size=in_qlayer, **qlayer_kwargs)

@LayerFactory.layer("hpvql")
class HPVQL(HVQL):
    """Classifier combining a linear layer, parallel VQC, and final linear head."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        in_qlayer: int,
        quantum_config: Mapping[str, Any] | QuantumDeviceConfig | None = None,
        **qlayer_kwargs,
    ) -> None:
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            in_qlayer=in_qlayer,
            quantum_config=quantum_config,
            **qlayer_kwargs,
        )

    def _qcls(self, in_qlayer, **qlayer_kwargs) -> Module:
        qlayer_kwargs['n_layers'] = qlayer_kwargs.pop("n_qdepth", None)
        return ParaVQC(input_size=in_qlayer, **qlayer_kwargs)

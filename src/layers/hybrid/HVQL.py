from __future__ import annotations

from torch import Tensor
from torch.nn import BatchNorm1d, Linear, LogSoftmax, Module, ReLU

from layers import BaseLayer, LayerFactory
from layers.quantum import VQC

from torch.nn import Module

from layers import LayerFactory
from .HVQL import HVQL
from layers.quantum import ParaVQC


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
        self.vqc = self._init_qlayer(input_size=in_qlayer, **qlayer_kwargs)
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
        n_layers: int | None = None,
        n_qdepth: int | None = None,
        embedding: str = "amplitude",
        ansatz: dict | None = None,
        **unused_kwargs,
    ) -> Module:
        n_layers = self._resolve_n_layers(n_layers, n_qdepth)

        if unused_kwargs:
            unexpected = ", ".join(sorted(unused_kwargs))
            raise TypeError(f"Unexpected keyword arguments for HVQL: {unexpected}")

        return VQC(
            input_size=input_size,
            n_layers=n_layers,
            embedding=embedding,
            ansatz=ansatz,
        )

    @staticmethod
    def _resolve_n_layers(
        n_layers: int | None,
        n_qdepth: int | None,
        *,
        default: int = 3,
    ) -> int:
        if n_layers is None:
            n_layers = n_qdepth if n_qdepth is not None else default
        elif n_qdepth is not None and n_layers != n_qdepth:
            raise ValueError(
                "Received conflicting values for 'n_layers' and 'n_qdepth'. "
                "Please provide only one or ensure they match."
            )

        if n_layers < 1:
            raise ValueError("Number of layers must be a positive integer")
        return n_layers


@LayerFactory.classifier("hpvql")
class HPVQL(HVQL):
    """Classifier combining a linear layer, parallel VQC, and final linear head."""

    def __init__(
        self,
        in_features: int,
        output_size: int,
        in_qlayer: int,
        n_qblocks: int,
        **qlayer_kwargs,
    ) -> None:
        qlayer_kwargs.setdefault("embedding", "angle")
        super().__init__(
            in_features=in_features,
            output_size=output_size,
            in_qlayer=in_qlayer,
            n_qblocks=n_qblocks,
            **qlayer_kwargs,
        )

    def _init_qlayer(
        self,
        *,
        input_size: int,
        n_layers: int | None = None,
        n_qdepth: int | None = None,
        embedding: str = "angle",
        ansatz: dict | None = None,
        n_qblocks: int | None = None,
        **unused_kwargs,
    ) -> Module:
        n_layers = self._resolve_n_layers(n_layers, n_qdepth)

        if n_qblocks is None:
            raise TypeError("HPVQL requires the 'n_qblocks' keyword argument")

        if n_qblocks < 1:
            raise ValueError("n_qblocks must be a positive integer")

        if unused_kwargs:
            unexpected = ", ".join(sorted(unused_kwargs))
            raise TypeError(f"Unexpected keyword arguments for HPVQL: {unexpected}")
        return ParallelVQC(
            input_size=input_size,
            n_qblocks=n_qblocks,
            n_layers=n_layers,
            ansatz=ansatz,
            embedding=embedding,
        )

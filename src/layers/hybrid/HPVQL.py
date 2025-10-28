from __future__ import annotations

from torch.nn import Module

from layers import LayerFactory
from .HVQL import HVQL
from layers.quantum import ParallelVQC

@LayerFactory.classifier("hpvql")
class HPVQL(HVQL):
    """Classifier combining a linear layer, parallel VQC, and final linear head."""

    def __init__(
        self,
        in_features: int,
        output_size: int,
        in_qlayer: int,
        n_qblocks: int,
        n_qdepth: int = 3,
        ansatz: dict | None = None,
        embedding: str = "angle",
    ) -> None:
        super().__init__(
            in_features=in_features,
            output_size=output_size,
            in_qlayer=in_qlayer,
            n_qdepth=n_qdepth,
            ansatz=ansatz,
            embedding=embedding,
            n_qblocks=n_qblocks,
        )

    def _init_qlayer(
        self,
        *,
        input_size: int,
        n_layers: int,
        embedding: str,
        ansatz: dict | None,
        n_qblocks: int,
        **unused_kwargs,
    ) -> Module:
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

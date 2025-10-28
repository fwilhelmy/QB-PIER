from __future__ import annotations

from torch import Tensor

from layers import BaseLayer, LayerFactory
from layers.quantum import ParallelVQC, VQC


class _BaseVQL(BaseLayer):
    """Common utilities for pure quantum classifier layers."""

    def __init__(self, *, in_features: int) -> None:
        super().__init__(in_features=in_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.vqc(x)


@LayerFactory.classifier("vqc")
class VQL(_BaseVQL):
    """Expose a single variational quantum circuit as a classifier layer."""

    def __init__(
        self,
        in_features: int,
        n_qdepth: int = 3,
        ansatz: dict | None = None,
        embedding: str = "angle",
    ) -> None:
        super().__init__(in_features=in_features)
        self.vqc = VQC(
            input_size=in_features,
            n_layers=n_qdepth,
            embedding=embedding,
            ansatz=ansatz,
        )
        self.output_size = self.vqc.output_size


@LayerFactory.classifier("parallel_vqc")
class ParallelVQL(_BaseVQL):
    """Expose a parallel VQC stack as a classifier layer."""

    def __init__(
        self,
        in_features: int,
        n_qblocks: int,
        n_qdepth: int = 3,
        ansatz: dict | None = None,
        embedding: str = "angle",
    ) -> None:
        super().__init__(in_features=in_features)
        self.vqc = ParallelVQC(
            input_size=in_features,
            n_qblocks=n_qblocks,
            n_layers=n_qdepth,
            embedding=embedding,
            ansatz=ansatz,
        )
        self.output_size = self.vqc.output_size


__all__ = ["VQL", "ParallelVQL"]

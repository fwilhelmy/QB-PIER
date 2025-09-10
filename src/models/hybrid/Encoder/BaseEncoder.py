from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor

from ..HybridLayer import HybridLayer


class BaseEncoder(HybridLayer, ABC):
    """Abstract base encoder for hybrid models."""

    def __init__(self, input_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.input_size = input_size

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Number of features produced by this encoder."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Encode an input batch into features."""
        raise NotImplementedError

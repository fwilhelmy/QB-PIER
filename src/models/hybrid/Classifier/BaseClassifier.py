from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module


class BaseClassifier(Module, ABC):
    """Abstract base classifier for hybrid models."""

    def __init__(self, in_features: int, output_size: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.output_size = output_size

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Compute class scores from encoded features."""
        raise NotImplementedError

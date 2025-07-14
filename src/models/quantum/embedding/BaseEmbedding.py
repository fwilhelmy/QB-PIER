"""Base classes for quantum data embeddings."""

from __future__ import annotations
from abc import ABC, abstractmethod

class BaseEmbedding(ABC):
    """Base class for all embedding strategies."""

    def __init__(self, input_size: int) -> None:
        self.input_size = input_size

    @property
    @abstractmethod
    def n_qubits(self) -> int:
        """Number of qubits required for this embedding."""
        raise NotImplementedError("Subclasses must implement n_qubits property.")

    @abstractmethod
    def __call__(self, inputs) -> None:
        """Apply the embedding. Default implementation does nothing."""
        raise NotImplementedError("Subclasses must implement __call__ method.")
    
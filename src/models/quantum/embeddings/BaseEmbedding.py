"""Base classes for quantum data embeddings."""

from __future__ import annotations

import pennylane as qml


class BaseEmbedding:
    """Base class for all embedding strategies."""

    def __init__(self, n_qubits: int, rotation: str = "X") -> None:
        self.n_qubits = n_qubits
        self.rotation = rotation

    def __call__(self, inputs) -> None:  # pragma: no cover - simple pass-through
        """Apply the embedding. Default implementation does nothing."""
        return None
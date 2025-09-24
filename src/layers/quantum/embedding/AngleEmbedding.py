"""Angle embedding strategy."""

from __future__ import annotations

import pennylane as qml

from . import BaseEmbedding, EmbeddingFactory

@EmbeddingFactory.embedding("angle")
class AngleEmbedding(BaseEmbedding):
    """Embed inputs as rotation angles."""

    def __init__(self, input_size: int, rotation: str = "X") -> None:
        super().__init__(input_size)
        self.rotation = rotation

    @property
    def n_qubits(self) -> int:
        """Number of qubits required for angle embedding."""
        return self.input_size

    def __call__(self, inputs) -> None:
        qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation=self.rotation)

"""Angle embedding strategy."""

from __future__ import annotations

import pennylane as qml

from models.quantum.embeddings import BaseEmbedding


class AngleEmbedding(BaseEmbedding):
    """Embed inputs as rotation angles."""

    def __call__(self, inputs) -> None:
        qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation=self.rotation)

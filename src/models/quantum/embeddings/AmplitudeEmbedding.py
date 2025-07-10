"""Amplitude embedding strategy."""

from __future__ import annotations

import pennylane as qml
import math

from models.quantum.embeddings import BaseEmbedding


class AmplitudeEmbedding(BaseEmbedding):
    """Embed inputs into the quantum state amplitudes."""

    def __init__(self, input_size: int) -> None:
        super().__init__(input_size)

    @property
    def n_qubits(self) -> int:
        """Number of qubits required for amplitude embedding."""
        return math.ceil(math.log2(self.input_size))

    def __call__(self, inputs) -> None:
        qml.AmplitudeEmbedding(
            inputs,
            wires=range(self.n_qubits),
            normalize=True,
            pad_with=0.0,
        )

"""Amplitude embedding strategy."""

from __future__ import annotations

import pennylane as qml

from models.quantum.embeddings import BaseEmbedding


class AmplitudeEmbedding(BaseEmbedding):
    """Embed inputs into the quantum state amplitudes."""

    def __call__(self, inputs) -> None:
        qml.AmplitudeEmbedding(
            inputs,
            wires=range(self.n_qubits),
            normalize=True,
            pad_with=0.0,
        )

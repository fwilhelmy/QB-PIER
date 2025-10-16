"""Basic smoke test for :mod:`layers.quantum.embedding.AmplitudeEmbedding`."""

from __future__ import annotations

import pytest

pytest.importorskip("pennylane")

from layers.quantum.embedding import AmplitudeEmbedding


def test_amplitude_embedding_rounds_qubit_count_up() -> None:
    embedding = AmplitudeEmbedding(input_size=3)
    assert embedding.n_qubits == 2

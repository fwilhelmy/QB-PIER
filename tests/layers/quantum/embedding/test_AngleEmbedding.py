"""Basic smoke test for :mod:`layers.quantum.embedding.AngleEmbedding`."""

from __future__ import annotations

import pytest

pytest.importorskip("pennylane")

from layers.quantum.embedding import AngleEmbedding


def test_angle_embedding_qubits_match_input_size() -> None:
    embedding = AngleEmbedding(input_size=3)
    assert embedding.n_qubits == 3

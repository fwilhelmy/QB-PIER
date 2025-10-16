"""Basic smoke test for :mod:`layers.quantum.embedding.EmbeddingFactory`."""

from __future__ import annotations

import pytest

pytest.importorskip("pennylane")

from layers.quantum.embedding import AngleEmbedding, EmbeddingFactory


def test_embedding_factory_creates_angle_embedding() -> None:
    embedding = EmbeddingFactory.create("angle", input_size=2)
    assert isinstance(embedding, AngleEmbedding)

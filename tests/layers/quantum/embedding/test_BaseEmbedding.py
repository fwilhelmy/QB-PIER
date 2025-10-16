"""Basic smoke test for :mod:`layers.quantum.embedding.BaseEmbedding`."""

from __future__ import annotations

from layers.quantum.embedding import BaseEmbedding


class _DummyEmbedding(BaseEmbedding):
    @property
    def n_qubits(self) -> int:  # pragma: no cover - trivial property
        return self.input_size

    def __call__(self, inputs):  # pragma: no cover - no-op
        return inputs


def test_base_embedding_preserves_input_size() -> None:
    emb = _DummyEmbedding(input_size=3)
    assert emb.input_size == 3
    assert emb.n_qubits == 3

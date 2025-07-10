"""Factory for creating embedding classes."""

from __future__ import annotations

from models.quantum.embeddings import BaseEmbedding, AngleEmbedding, AmplitudeEmbedding


def EmbeddingFactory(name: str, input_size: int, rotation: str = "X") -> tuple[BaseEmbedding | None, int]:
    """Instantiate embedding objects from a string name."""
    name = name.lower()
    if name == "angle":
        embedding = AngleEmbedding(input_size, rotation)
        return embedding, embedding.n_qubits
    if name == "amplitude":
        embedding = AmplitudeEmbedding(input_size)
        return embedding, embedding.n_qubits
    if name == "none":
        return None, input_size  # No embedding, just return the number of qubits
    raise ValueError(f"Unsupported embedding: {name}")

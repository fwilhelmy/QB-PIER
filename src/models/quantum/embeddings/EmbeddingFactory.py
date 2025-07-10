from __future__ import annotations

"""Factory for creating embedding classes."""

from typing import Type

from models.quantum.embeddings import BaseEmbedding, AngleEmbedding, AmplitudeEmbedding


class EmbeddingFactory:
    """Instantiate embedding objects from a string name."""

    @staticmethod
    def create(name: str, n_qubits: int, rotation: str = "X") -> BaseEmbedding:
        name = name.lower()
        if name == "angle":
            return AngleEmbedding(n_qubits, rotation)
        if name == "amplitude":
            return AmplitudeEmbedding(n_qubits, rotation)
        if name == "none":
            return BaseEmbedding(n_qubits, rotation)
        raise ValueError(f"Unsupported embedding: {name}")

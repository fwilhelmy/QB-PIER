import pennylane as qml
from src.models.quantum.ansatzs.BaseAnsatz import BaseAnsatz
import numpy as np

class IdentityAnsatz(BaseAnsatz):
    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 0,
        embedding: str = "angle",
        rotation: str = "X",
    ):
        super().__init__(n_qubits, n_layers, embedding, rotation)
        self.weight_shapes = {"weights": (0,)} # No trainable parameters

    def ansatz(self, weights):
        pass # Intentionally no entangling layers
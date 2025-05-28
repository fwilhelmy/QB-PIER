import pennylane as qml
from models.quantum.ansatzs.BaseAnsatz import BaseAnsatz
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
        # No trainable parameters
        self.weight_shapes = {"weights": (0,)}

    def ansatz(self, weights):
        # Intentionally no entangling layers
        pass

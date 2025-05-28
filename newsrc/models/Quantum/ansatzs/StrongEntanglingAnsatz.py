import pennylane as qml
from models.quantum.ansatzs.BaseAnsatz import BaseAnsatz
import numpy as np

class StrongEntanglingAnsatz(BaseAnsatz):
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        embedding: str = "angle",
        rotation: str = "X",
    ):
        super().__init__(n_qubits, n_layers, embedding, rotation)
        # PennyLane StronglyEntanglingLayers expects weights of shape (n_layers, n_qubits, 3)
        # self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.weight_shapes = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        print(f"StrongEntanglingAnsatz: weight_shapes = {self.weight_shapes}")

    def ansatz(self, weights):
        qml.StronglyEntanglingLayers(
            weights,
            wires=range(self.n_qubits),
            ranges=np.ones(3, dtype=int),
        )
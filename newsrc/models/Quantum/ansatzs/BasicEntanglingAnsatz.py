import pennylane as qml
from models.quantum.ansatzs.BaseAnsatz import BaseAnsatz
import numpy as np

class BasicEntanglingAnsatz(BaseAnsatz):
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        embedding: str = "angle",
        rotation: str = "X",
    ):
        super().__init__(n_qubits, n_layers, embedding, rotation)
        # PennyLane BasicEntanglerLayers expects weights of shape (n_layers, n_qubits)
        # self.weight_shapes = {"weights": (n_layers, n_qubits)}
        self.weight_shapes = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        print(f"BasicEntanglingAnsatz: weight_shapes = {self.weight_shapes}")

    def ansatz(self, weights):
        qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
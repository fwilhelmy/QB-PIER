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
        
        self.weight_shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)}

    def ansatz(self, weights):
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
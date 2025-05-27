import pennylane as qml
from layers.ansatzs.BaseAnsatz import BaseAnsatz
import numpy as np

class RandomStructureAnsatz(BaseAnsatz):
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        embedding: str = "angle",
        rotation: str = "Y",
        ratio: float = 0.75,
    ):
        super().__init__(n_qubits, n_layers, embedding, rotation)
        self.weight_shapes = {"weights": (n_layers, n_qubits)}
        self.ratio = ratio

    def ansatz(self, weights):
        qml.RandomLayers(
            weights,
            wires=range(self.n_qubits),
            ratio_imprim=self.ratio,
        )
import pennylane as qml
from models.quantum.ansatzs.BaseAnsatz import BaseAnsatz
import numpy as np

class RandomStructureAnsatz(BaseAnsatz):
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        embedding: str = "angle",
        rotation: str = "Y",
        ratio: float = 0.75, # Ratio of rotation 1 qubit gates to cnots
    ):
        super().__init__(n_qubits, n_layers, embedding, rotation)

        self.weight_shapes = {"weights": qml.RandomLayers.shape(n_layers=n_layers, n_rotations=3)}
        self.ratio = ratio

    def ansatz(self, weights):
        qml.RandomLayers(
            weights,
            wires=range(self.n_qubits),
            ratio_imprim=self.ratio,
        )
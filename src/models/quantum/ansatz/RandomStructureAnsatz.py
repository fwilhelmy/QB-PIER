import pennylane as qml
from models.quantum.ansatz.BaseAnsatz import BaseAnsatz

class RandomStructureAnsatz(BaseAnsatz):
    def __init__(
        self,
        input_size: int,
        n_layers: int,
        embedding: str = "angle",
        ratio_imprim: float = 0.75, # Ratio of rotation 1 qubit gates to cnots
    ):
        super().__init__(input_size, n_layers, embedding)
        self.ratio_imprim = ratio_imprim

    @property
    def weight_shapes(self) -> dict:
        return {"weights": qml.RandomLayers.shape(n_layers=self.n_layers, n_rotations=3)}

    def ansatz(self, weights):
        qml.RandomLayers(
            weights,
            wires=range(self.n_qubits),
            ratio_imprim=self.ratio_imprim,
        )
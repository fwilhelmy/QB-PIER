import pennylane as qml
from models.quantum.ansatzs.BaseAnsatz import BaseAnsatz

class BasicEntanglingAnsatz(BaseAnsatz):
    def __init__(
        self,
        input_size: int,
        n_layers: int,
        embedding: str = "angle"
    ):
        super().__init__(input_size, n_layers, embedding)

    @property
    def weight_shapes(self) -> dict:
        """
        Define the shape of the weights for the Basic Entangling Ansatz.
        Returns a dictionary with weight names and their shapes.
        """
        return {"weights": qml.BasicEntanglerLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)}

    def ansatz(self, weights):
        qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
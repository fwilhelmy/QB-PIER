import pennylane as qml
from models.quantum.ansatz.BaseAnsatz import BaseAnsatz
from models.quantum.ansatz.AnsatzFactory import AnsatzFactory

@AnsatzFactory.ansatz("NoEntanglingAnsatz")
class NoEntanglingAnsatz(BaseAnsatz):
    def __init__(
        self,
        input_size: int,
        n_layers: int,
        embedding: str = "angle"
    ):
        super().__init__(input_size, n_layers, embedding)

    @property
    def weight_shapes(self) -> dict:
        return {"weights": (self.n_layers, self.n_qubits, 3)}  # Three angles per qubit in each layer

    def ansatz(self, weights):
        # Apply rotations layer by layer
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.Rot(*weights[layer, i], wires=i)

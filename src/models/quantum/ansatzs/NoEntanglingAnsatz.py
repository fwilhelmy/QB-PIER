import pennylane as qml
from models.quantum.ansatzs.BaseAnsatz import BaseAnsatz

class NoEntanglingAnsatz(BaseAnsatz):
    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 0,
        embedding: str = "angle",
        rotation: str = "X",
    ):
        super().__init__(n_qubits, n_layers, embedding, rotation)
        # Single-qubit rotations only: one parameter per qubit
        self.weight_shapes = {"weights": (n_qubits,)}

    def ansatz(self, weights):
        # Apply a rotation on each qubit with its corresponding weight
        for i in range(self.n_qubits):
            qml.RY(weights[i], wires=i) # TODO CHANGE TO ROTATION TYPE
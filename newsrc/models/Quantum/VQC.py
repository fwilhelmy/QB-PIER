from layers.ansatzs.BasicEntanglingAnsatz import BasicEntanglingAnsatz
import pennylane as qml
import torch
import torch.nn as nn
from utils import create_qtorchlayer

class VQC(nn.Module):
    """
    Single quantum circuit module (Variational Quantum Circuit).
    Requires size_in equal to n_qubits.
    """
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        ansatz_cls=BasicEntanglingAnsatz,
        embedding: str = "angle",
        rotation: str = "X"
    ):
        super().__init__()
        self.n_qubits = n_qubits # Input size must match number of qubits
        self.n_layers = n_layers

        # Instantiate ansatz and create QNode layer
        self.ansatz = ansatz_cls(n_qubits, n_layers, embedding, rotation)
        self.qlayer = create_qtorchlayer(self.ansatz.n_qubits, self.ansatz.circuit, self.ansatz.weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expect x of shape (batch_size, n_qubits).
        Returns tensor of shape (batch_size, n_qubits).
        """
        return self.qlayer(x)
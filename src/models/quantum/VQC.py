from src.models.quantum.ansatzs.BasicEntanglingAnsatz import BasicEntanglingAnsatz
from src.models.quantum.ansatzs.BaseAnsatz import BaseAnsatz
import pennylane as qml
import torch
import torch.nn as nn
from src.utils import create_qtorchlayer
from typing import Type

class VQC(nn.Module):
    """
    Single quantum circuit module (Variational Quantum Circuit).
    Requires size_in equal to n_qubits.
    """
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        ansatz_cls: Type[BaseAnsatz] = BasicEntanglingAnsatz,
        embedding: str = "angle",
        rotation: str = "X"
    ):
        super().__init__()
        self.n_qubits = n_qubits # Input size must match number of qubits
        self.n_layers = n_layers

        # Instantiate ansatz and create QNode layer
        self.ansatz = ansatz_cls(n_qubits, n_layers, embedding, rotation)
        self.qlayer = self.ansatz.build_qlayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expect x of shape (batch_size, n_qubits).
        Returns tensor of shape (batch_size, n_qubits).
        """
        return self.qlayer(x)
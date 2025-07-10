from models.quantum.ansatz.BasicEntanglingAnsatz import BasicEntanglingAnsatz
from models.quantum.ansatz.BaseAnsatz import BaseAnsatz
import torch
import torch.nn as nn
from typing import Type

class VQC(nn.Module):
    """
    Single quantum circuit module (Variational Quantum Circuit).
    Requires size_in equal to n_qubits.
    """
    def __init__(
        self,
        input_size: int,
        n_layers: int,
        ansatz_cls: Type[BaseAnsatz] = BasicEntanglingAnsatz,
        embedding: str = "angle"
    ):
        super().__init__()
        self.input_size = input_size

        # Instantiate ansatz and create QNode layer
        self.ansatz = ansatz_cls(input_size, n_layers, embedding)
        self.qlayer = self.ansatz.build_qlayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expect x of shape (batch_size, n_qubits).
        Returns tensor of shape (batch_size, n_qubits).
        """
        return self.qlayer(x)
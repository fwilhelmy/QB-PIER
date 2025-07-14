from models.quantum.ansatz.BasicEntanglingAnsatz import BasicEntanglingAnsatz
from models.quantum.ansatz.BaseAnsatz import BaseAnsatz
from models.quantum.ansatz.AnsatzFactory import AnsatzFactory
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
        embedding: str = "angle",
        ansatz_kwargs: dict | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_qubits = input_size

        if ansatz_kwargs is None:
            ansatz_kwargs = {}

        ansatz_cls = ansatz_kwargs.pop("ansatz_cls", BasicEntanglingAnsatz)
        factory = AnsatzFactory(ansatz_cls, **ansatz_kwargs)

        # Instantiate ansatz and create QNode layer
        self.ansatz = factory.build(input_size, n_layers, embedding)
        self.qlayer = self.ansatz.build_qlayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expect x of shape (batch_size, n_qubits).
        Returns tensor of shape (batch_size, n_qubits).
        """
        return self.qlayer(x)
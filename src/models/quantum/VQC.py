from models.quantum.ansatz.AnsatzFactory import AnsatzFactory
import torch
import torch.nn as nn

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
        ansatz: dict | None = None,
    ):
        super().__init__()
        self.ansatz = AnsatzFactory(input_size, n_layers, embedding, ansatz)
        self.qlayer = self.ansatz.qlayer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expect x of shape (batch_size, n_qubits).
        Returns tensor of shape (batch_size, n_qubits).
        """
        return self.qlayer(x)
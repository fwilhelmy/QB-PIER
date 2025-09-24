from .ansatz import AnsatzFactory
import torch
import torch.nn as nn

class VQC(nn.Module):
    """Single quantum circuit module (Variational Quantum Circuit).

    Requires ``size_in`` equal to ``n_qubits``.
    """

    def __init__(
        self,
        input_size: int,
        n_layers: int,
        embedding: str = "angle",
        ansatz: dict | None = None,
    ):
        super().__init__()
        ansatz = ansatz or {}
        ansatz_name = ansatz.get("class", "StrongEntanglingAnsatz")
        ansatz_config = ansatz.get("params", {})
        self.ansatz = AnsatzFactory.create(
            ansatz_name,
            input_size=input_size,
            n_layers=n_layers,
            embedding=embedding,
            **ansatz_config,
        )
        self.qlayer = self.ansatz.qlayer

    @property
    def output_size(self) -> int:
        return self.ansatz.n_qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expect ``x`` of shape ``(batch_size, n_qubits)``.

        Returns tensor of shape ``(batch_size, n_qubits)``.
        """

        return self.qlayer(x)


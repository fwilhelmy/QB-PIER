from .ansatz import AnsatzFactory
import torch
import torch.nn as nn
from torch import Tensor

from layers import BaseLayer, LayerFactory

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

@LayerFactory.layer("vql")
class VQL(BaseLayer):
    """Expose a single variational quantum circuit as a classifier layer."""

    def __init__(
        self,
        input_size: int,
        n_qdepth: int = 3,
        ansatz: dict | None = None,
        embedding: str = "angle",
        output_size: int | None = None,
        apply_log_softmax: bool = True,
    ) -> None:
        super().__init__(input_size=input_size)
        _ = apply_log_softmax  # For API compatibility; activation handled upstream.
        self.vqc = VQC(
            input_size=input_size,
            n_layers=n_qdepth,
            embedding=embedding,
            ansatz=ansatz,
        )
        self.output_size = self.vqc.output_size

    def forward(self, x: Tensor) -> Tensor:
        return self.vqc(x)

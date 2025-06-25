import pennylane as qml
from src.models.quantum.ansatzs.BasicEntanglingAnsatz import BasicEntanglingAnsatz
from src.models.quantum.ansatzs.BaseAnsatz import BaseAnsatz
import torch
import torch.nn as nn
from typing import Type
import numpy as np
from src.models.quantum.VQC import VQC

class ParallelVQC(nn.Module):
    """
    Stacks multiple VQC modules to handle inputs larger than a single circuit.
    """
    def __init__(
        self,
        size_in: int,
        n_qubits: int,
        n_layers: int,
        ansatz_cls: Type[BaseAnsatz] = BasicEntanglingAnsatz,
        embedding: str = "angle",
        rotation: str = "X"
    ):
        super().__init__()
        self.size_in = size_in
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Number of VQC blocks needed
        self.num_blocks = int(np.ceil(size_in / n_qubits))

        # Create VQC for each block
        self.blocks = nn.ModuleList([
            VQC(
                n_qubits=n_qubits,
                n_layers=n_layers,
                ansatz_cls=ansatz_cls,
                embedding=embedding,
                rotation=rotation
            ) for _ in range(self.num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits x into chunks of size n_qubits, applies each VQC, and concatenates outputs.
        """
        # Split along feature dimension
        chunks = torch.split(x, self.n_qubits, dim=1)
        outputs = []
        for block, chunk in zip(self.blocks, chunks):
            # Pad last chunk if smaller than n_qubits
            if chunk.shape[1] < self.n_qubits:
                pad = torch.zeros(
                    chunk.size(0), self.n_qubits - chunk.size(1),
                    device=chunk.device, dtype=chunk.dtype
                )
                chunk = torch.cat([chunk, pad], dim=1)
            outputs.append(block(chunk))
        return torch.cat(outputs, dim=1)
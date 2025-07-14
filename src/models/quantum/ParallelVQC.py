from models.quantum.ansatz.BasicEntanglingAnsatz import BasicEntanglingAnsatz
from models.quantum.ansatz.BaseAnsatz import BaseAnsatz
import torch
import torch.nn as nn
from typing import Type
import numpy as np
from models.quantum.VQC import VQC

class ParallelVQC(nn.Module):
    """
    Stacks multiple VQC modules to handle inputs larger than a single circuit.
    """
    def __init__(
        self,
        input_size: int,
        block_size: int,
        n_layers: int,
        embedding: str = "angle",
        ansatz_kwargs: dict | None = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.block_size = block_size
        self.n_qubits = block_size
        self.num_blocks = int(np.ceil(input_size / block_size))
        self.n_layers = n_layers

        if ansatz_kwargs is None:
            ansatz_kwargs = {}

        # Create VQC for each block
        self.blocks = nn.ModuleList([
            VQC(
                input_size=block_size,
                n_layers=n_layers,
                embedding=embedding,
                ansatz_kwargs=ansatz_kwargs,
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
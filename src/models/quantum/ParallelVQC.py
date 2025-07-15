from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
from models.quantum.VQC import VQC

class ParallelVQC(nn.Module):
    """
    Stacks multiple VQC modules to handle inputs larger than a single circuit.
    """
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        block_size: int,
        n_layers: int,
        embedding: str = "angle",
        ansatz: dict | None = None,
    ):
        super().__init__()

        self.input_size = np.prod(self.input_size, dtype=int)
        self.block_size = min(block_size, self.input_size)
        self.n_blocks = int(np.ceil(self.input_size / self.block_size))
        self.n_layers = n_layers

        # Create VQC for each block
        self.blocks = nn.ModuleList([
            VQC(
                input_size=block_size,
                n_layers=n_layers,
                embedding=embedding,
                ansatz=ansatz,
            ) for _ in range(self.n_blocks)
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
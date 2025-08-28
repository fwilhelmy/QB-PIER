"""Utilities for stacking multiple variational quantum circuits."""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from models.quantum.VQC import VQC


class ParallelVQC(nn.Module):
    """Stacks multiple :class:`VQC` blocks to process larger inputs.

    The input feature vector is split into chunks of ``block_size`` elements.
    Each chunk is processed by its own :class:`VQC` instance and the results are
    concatenated.
    """

    def __init__(
        self,
        input_size: Tuple[int, int, int] | int,
        block_size: int,
        n_layers: int,
        embedding: str = "angle",
        ansatz: dict | None = None,
    ) -> None:
        super().__init__()

        # Flatten the input size to a single integer representing the feature
        # vector that will be split across the quantum blocks.
        self.input_size = int(np.prod(input_size))
        self.block_size = min(block_size, self.input_size)
        self.n_blocks = int(np.ceil(self.input_size / self.block_size))
        self.n_layers = n_layers

        # Expose the size of the concatenated outputs so that classical layers
        # can be configured accordingly.
        self.output_size = self.block_size * self.n_blocks

        # Create VQC for each block
        self.blocks = nn.ModuleList(
            [
                VQC(
                    input_size=self.block_size,
                    n_layers=n_layers,
                    embedding=embedding,
                    ansatz=ansatz,
                )
                for _ in range(self.n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all quantum blocks in parallel and concatenate the results."""

        chunks = torch.split(x, self.block_size, dim=1)
        outputs = []
        for block, chunk in zip(self.blocks, chunks):
            if chunk.shape[1] < self.block_size:
                pad = torch.zeros(
                    chunk.size(0),
                    self.block_size - chunk.size(1),
                    device=chunk.device,
                    dtype=chunk.dtype,
                )
                chunk = torch.cat([chunk, pad], dim=1)
            outputs.append(block(chunk))
        return torch.cat(outputs, dim=1)


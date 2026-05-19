"""Utilities for stacking multiple variational quantum circuits."""

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..BaseLayer import BaseLayer
from ..LayerFactory import LayerFactory
from .VQC import VQC
from qdevice import QuantumDeviceConfig

class ParaVQC(nn.Module):
    """Stacks multiple :class:`VQC` blocks to process larger inputs.

    The input feature vector is split into ``n_qblocks`` chunks with comparable
    sizes.  Each chunk is processed by its own :class:`VQC` instance and the
    results are concatenated.
    """

    def __init__(
        self,
        input_size: int,
        n_qblocks: int,
        n_layers: int,
        embedding: str = "angle",
        ansatz: dict | None = None,
        quantum_config: Mapping[str, Any] | QuantumDeviceConfig | None = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size

        # Restrict the number of blocks to the available number of features so
        # that no empty quantum circuits are created.
        self.n_blocks = min(int(n_qblocks), self.input_size)
        self.block_size = int(np.ceil(self.input_size / self.n_blocks))
        self.n_layers = n_layers

        # Create VQC for each block
        self.blocks = nn.ModuleList(
            [
                VQC(
                    input_size=self.block_size,
                    n_layers=n_layers,
                    embedding=embedding,
                    ansatz=ansatz,
                    quantum_config=quantum_config,
                )
                for _ in range(self.n_blocks)
            ]
        )

        # Expose the size of the concatenated outputs so that classical layers
        # can be configured accordingly.
        self.output_size = sum(block.output_size for block in self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all quantum blocks in parallel and concatenate the results."""

        chunks = torch.split(x, self.block_size, dim=1)
        outputs = []
        batch_size, device, dtype = x.size(0), x.device, x.dtype
        for idx, block in enumerate(self.blocks):
            if idx < len(chunks):
                chunk = chunks[idx]
            else:
                chunk = torch.zeros(batch_size, self.block_size, device=device, dtype=dtype)
            if chunk.shape[1] < self.block_size:
                pad = torch.zeros(
                    batch_size,
                    self.block_size - chunk.size(1),
                    device=device,
                    dtype=dtype,
                )
                chunk = torch.cat([chunk, pad], dim=1)
            outputs.append(block(chunk))
        return torch.cat(outputs, dim=1)

@LayerFactory.layer("pvql")
class ParaVQL(BaseLayer):
    """Expose a parallel VQC stack as a classifier layer."""

    def __init__(
        self,
        input_size: int,
        n_qblocks: int,
        n_qdepth: int = 3,
        ansatz: dict | None = None,
        embedding: str = "angle",
        quantum_config: Mapping[str, Any] | QuantumDeviceConfig | None = None,
    ) -> None:
        super().__init__(input_size=input_size)

        self.vqc = ParaVQC(
            input_size=input_size,
            n_qblocks=n_qblocks,
            n_layers=n_qdepth,
            embedding=embedding,
            ansatz=ansatz,
            quantum_config=quantum_config,
        )

        self.output_size = self.vqc.output_size

    def forward(self, x: Tensor) -> Tensor:
        return self.vqc(x)

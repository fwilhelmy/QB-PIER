"""Variational quantum classifier model built on top of :class:`VQC`."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .. import BaseModel, ModelFactory
from ...layers.quantum import VQC

@ModelFactory.model("ampvqc") # Legacy name for backward compatibility
@ModelFactory.model("vqm")
class VQM(BaseModel):
    """Variational Quantum Model (VQM) for variational quantum classification.

    The VQM wraps :class:`models.quantum.VQC` while exposing configuration knobs
    for the quantum embedding, ansatz definition, and circuit depth. It keeps the
    legacy ``ampvqc`` registry entry so existing experiment configurations can be
    reused without modification.
    """

    def __init__(
        self,
        n_qdepth: int = 1,
        ansatz: dict | None = None,
        embedding: str = "amplitude",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters("n_qdepth", "ansatz", "embedding")

        self.vqc = VQC(
            input_size=np.prod(self.input_size, dtype=int),
            n_layers=n_qdepth,
            embedding=embedding,
            ansatz=ansatz,
        )

        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.view(inputs.size(0), -1)
        logits = self.vqc(batch)
        return self.logSoftmax(logits)

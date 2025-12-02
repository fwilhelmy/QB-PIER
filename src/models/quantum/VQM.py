"""Variational quantum classifier model built on top of :class:`VQC`."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from models import BaseModel, ModelFactory
from layers.quantum import VQC

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

        if self.vqc.output_size != self.output_size:
            # Learned linear projection decouples features (LexGrouping keeps them on the simplex).
            self.post = nn.Linear(self.vqc.output_size, self.output_size, bias=True)
        else:
            self.post = nn.Identity()

        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.view(inputs.size(0), -1)
        logits = self.post(self.vqc(batch))
        return self.logSoftmax(logits)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        map_location: str | torch.device | None = None,
        strict: bool | None = None,
        **kwargs,
    ) -> "VQM":
        """Load a checkpoint while remaining compatible with older runs."""

        if strict is None:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            state_dict = checkpoint.get("state_dict", checkpoint)
            strict = any(key.startswith("post.") for key in state_dict)

        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            strict=strict,
            **kwargs,
        )

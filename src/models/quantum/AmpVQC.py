import torch
from torch.nn import LogSoftmax

import numpy as np
from models.quantum import VQC
from models import BaseModel
from metrics import LoggingService
from models import ModelFactory

@ModelFactory.model("ampvqc")
class AmpVQC(BaseModel):

    def __init__(
        self,
        n_qdepth: int = 1,
        ansatz: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters("n_qdepth", "ansatz")

        self.vqc = VQC(
            input_size=np.prod(self.input_size, dtype=int),
            n_layers=n_qdepth,
            embedding="amplitude",
            ansatz=ansatz,
        )

        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.view(inputs.size(0), -1)
        x = self.vqc(x)
        return self.logSoftmax(x)

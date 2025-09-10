import numpy as np
import torch
from torch import nn
from torch.nn import LogSoftmax
from models.quantum import VQC
from models import BaseModel, ModelFactory

@ModelFactory.model("hqe")
class HQE(BaseModel):
    """Hybrid Quantum Encoder (HQE) model

    Consists of classical encoder layers followed by a variational quantum
    circuit acting as the bottleneck. The number of qubits equals the number
    of output classes.
    """

    def __init__(self, hidden_sizes, n_qdepth: int = 1,
                 ansatz: dict | None = None,
                 embedding: str = "amplitude",
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters("hidden_sizes", "n_qdepth", "ansatz", "embedding")
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        layers = []
        prev = np.prod(self.input_size, dtype=int)
        for idx, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.encoder = nn.Sequential(*layers)

        # Compress to dimension equal to number of qubits/classes
        self.bottleneck = nn.Linear(prev, self.output_size)
        self.bottleneck_act = nn.ReLU()

        self.vqc = VQC(
            input_size=self.output_size,
            n_layers=n_qdepth,
            ansatz=ansatz,
            embedding=embedding,
        )
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(inputs, 1)
        x = self.encoder(x)
        x = self.bottleneck_act(self.bottleneck(x))
        x = self.vqc(x)
        return self.logSoftmax(x)
    
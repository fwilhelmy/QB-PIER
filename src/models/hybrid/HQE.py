import torch
from torch import nn
from torch.nn import LogSoftmax
from src.models.quantum.VQC import VQC
from src.models.quantum.ansatzs.BasicEntanglingAnsatz import BasicEntanglingAnsatz
from src.models.BaseModel import BaseModel


class HQE(BaseModel):
    """Hybrid Quantum Encoder (HQE) model

    Consists of classical encoder layers followed by a variational quantum
    circuit acting as the bottleneck. The number of qubits equals the number
    of output classes.
    """

    def __init__(self, input_size: int, hidden_sizes, classes: int,
                 n_qdepth: int = 1,
                 ansatz_cls=BasicEntanglingAnsatz,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        self.classes = classes
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        layers = []
        prev = input_size
        for idx, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.encoder = nn.Sequential(*layers)

        # Compress to dimension equal to number of qubits/classes
        self.bottleneck = nn.Linear(prev, classes)
        self.bottleneck_act = nn.ReLU()

        self.vqc = VQC(n_qubits=classes, n_layers=n_qdepth, ansatz_cls=ansatz_cls)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.view(inputs.size(0), -1)
        x = self.encoder(x)
        x = self.bottleneck_act(self.bottleneck(x))
        x = self.vqc(x)
        return self.logSoftmax(x)
    
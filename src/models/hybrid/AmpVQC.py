import torch
from torch import nn
from torch.nn import LogSoftmax

from models.DimensionalityReductionLayer import DimensionalityReductionLayer
from models.quantum.VQC import VQC
from models.quantum.ansatzs.BasicEntanglingAnsatz import BasicEntanglingAnsatz
from models.BaseModel import BaseModel
from typing import Type
import math

class AmpVQC(BaseModel):

    def __init__(
        self,
        input_size: int,
        n_qdepth: int = 1,
        ansatz_cls: Type[BasicEntanglingAnsatz] = BasicEntanglingAnsatz,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        assert "ResidualAnsatz" not in ansatz_cls.__name__, "Amplitude encoding does not support residual ansatzes."
        n_qubits = math.ceil(math.log2(input_size))
        self.vqc = VQC(n_qubits=n_qubits, n_layers=n_qdepth, ansatz_cls=ansatz_cls, embedding="amplitude")
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.view(inputs.size(0), -1)
        x = self.vqc(x)
        return self.logSoftmax(x)

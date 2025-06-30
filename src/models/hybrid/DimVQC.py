import torch
from torch import nn
from torch.nn import LogSoftmax

from models.DimensionalityReductionLayer import DimensionalityReductionLayer
from models.quantum.VQC import VQC
from models.quantum.ansatzs.BasicEntanglingAnsatz import BasicEntanglingAnsatz
from models.BaseModel import BaseModel
from typing import Type


class DimVQC(BaseModel):
    """Hybrid model applying dimensionality reduction followed by a VQC."""

    def __init__(
        self,
        input_size: int,
        reduction_method: str,
        reduced_dim: int,
        n_qdepth: int = 1,
        ansatz_cls: Type[BasicEntanglingAnsatz] = BasicEntanglingAnsatz,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        self.reduced_dim = reduced_dim

        self.reducer = DimensionalityReductionLayer(reduction_method, reduced_dim)
        self.vqc = VQC(n_qubits=reduced_dim, n_layers=n_qdepth, ansatz_cls=ansatz_cls)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.view(inputs.size(0), -1)
        x = self.reducer(x)
        x = self.vqc(x)
        return self.logSoftmax(x)

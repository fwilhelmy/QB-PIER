import torch
from torch.nn import LogSoftmax

from models.quantum.VQC import VQC
from models.quantum.ansatzs.BasicEntanglingAnsatz import BasicEntanglingAnsatz
from models.BaseModel import BaseModel
from typing import Type
from metrics import LoggingService
import math

class AmpVQC(BaseModel):

    def __init__(
        self,
        n_qdepth: int = 1,
        ansatz_cls: Type[BasicEntanglingAnsatz] = BasicEntanglingAnsatz,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters("n_qdepth", "ansatz_cls")
        # flatten input size to a single dimension
        self.flat_input_dim = self.input_size[0] * self.input_size[1] * self.input_size[2]
        assert "ResidualAnsatz" not in ansatz_cls.__name__, "Amplitude encoding does not support residual ansatzes."
        n_qubits = math.ceil(math.log2(self.flat_input_dim))
        self.vqc = VQC(n_qubits=n_qubits, n_layers=n_qdepth, ansatz_cls=ansatz_cls, embedding="amplitude")
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.view(inputs.size(0), -1)
        x = self.vqc(x)
        return self.logSoftmax(x)
    
    def on_fit_end(self) -> None:
        circuit_fig = self.vqc.ansatz.visualize(True)
        path = f"{self.__class__.__name__}/{self.model_name}"
        LoggingService().save_figure(
            circuit_fig,
            "circuit_diagram",
            path,
        )

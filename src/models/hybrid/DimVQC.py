import torch
from torch.nn import LogSoftmax

from models.classical.DimensionalityReductionLayer import DimensionalityReductionLayer
from models.quantum.VQC import VQC
from models.quantum.ansatz.BasicEntanglingAnsatz import BasicEntanglingAnsatz
from models.BaseModel import BaseModel
from metrics import LoggingService
import math


class DimVQC(BaseModel):
    """Hybrid model applying dimensionality reduction followed by a VQC."""

    def __init__(
        self,
        reduced_dim: int,
        reduction_method: str = "pca",
        n_qdepth: int = 1,
        ansatz: dict | None = None,
        embedding: str = "amplitude",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters("reduction_method", "reduced_dim", "n_qdepth", "ansatz")

        n_qubits = math.ceil(math.log2(reduced_dim))

        self.reducer = DimensionalityReductionLayer(reduction_method, reduced_dim)
        if ansatz is None:
            ansatz = {}
        self.vqc = VQC(
            input_size=n_qubits,
            n_layers=n_qdepth,
            embedding=embedding,
            ansatz=ansatz,
        )
        self.logSoftmax = LogSoftmax(dim=self.input_size[0])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.view(inputs.size(0), -1)
        x = self.reducer(x)
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
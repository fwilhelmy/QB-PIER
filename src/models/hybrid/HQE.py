import torch
from torch import nn
from torch.nn import LogSoftmax
from models.quantum.VQC import VQC
from models.quantum.ansatz.BasicEntanglingAnsatz import BasicEntanglingAnsatz
from models.BaseModel import BaseModel
from metrics import LoggingService


class HQE(BaseModel):
    """Hybrid Quantum Encoder (HQE) model

    Consists of classical encoder layers followed by a variational quantum
    circuit acting as the bottleneck. The number of qubits equals the number
    of output classes.
    """

    def __init__(self, hidden_sizes, n_qdepth: int = 1,
                 ansatz_kwargs: dict | None = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters("hidden_sizes", "n_qdepth", "ansatz_kwargs")
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
        self.bottleneck = nn.Linear(prev, self.output_size)
        self.bottleneck_act = nn.ReLU()

        if ansatz_kwargs is None:
            ansatz_kwargs = {}
        self.vqc = VQC(
            input_size=self.output_size,
            n_layers=n_qdepth,
            ansatz_kwargs=ansatz_kwargs,
        )
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.view(inputs.size(0), -1)
        x = self.encoder(x)
        x = self.bottleneck_act(self.bottleneck(x))
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
    
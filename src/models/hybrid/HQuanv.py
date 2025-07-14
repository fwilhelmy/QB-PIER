"""Hybrid CNN model using a quanvolution layer."""

from typing import Optional, Type

import torch
from torch import flatten, float32
from torch.nn import Linear, LogSoftmax, ReLU
from metrics import LoggingService

from models.BaseModel import BaseModel
from models.quantum.Quanv2d import Quanv2d
from models.quantum.ansatz.BaseAnsatz import BaseAnsatz
from models.quantum.ansatz.RandomStructureAnsatz import RandomStructureAnsatz

class HQuanv(BaseModel):
    """Hybrid Quanvolutional Neural Network."""

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 2,
        stride: Optional[int] = None,
        n_qdepth: int = 1,
        ansatz: dict | None = None,
        embedding: str = "angle",
        rotation: str = "X",
        fc_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters(
            "img_size",
            "patch_size",
            "stride",
            "n_qdepth",
            "ansatz",
            "embedding",
            "rotation",
            "fc_size",
        )

        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride or patch_size

        # Quantum convolution layer
        if ansatz is None:
            ansatz = {}
        self.quanv = Quanv2d(
            in_channels=1,
            out_channels=self.patch_size * self.patch_size,
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=0,
            dilation=1,
            n_layers=n_qdepth,
            embedding=embedding,
            ansatz=ansatz,
        )

        n_patches = (self.img_size - self.patch_size) // self.stride + 1
        self.fc_input_dim = fc_size or self.quanv.out_channels * n_patches * n_patches

        # Classical fully connected layers
        self.fc1 = Linear(in_features=self.fc_input_dim, out_features=self.fc_input_dim)
        self.relu3 = ReLU()

        self.fc2 = Linear(in_features=self.fc_input_dim, out_features=self.output_size)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Quantum convolution
        x = self.quanv(inputs)

        # Flatten and pass through classical layers
        x = flatten(x.to(float32), 1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        output = self.logSoftmax(x)

        return output
    
    def on_fit_end(self) -> None:
        circuit_fig = self.quanv.vqc.ansatz.visualize(True)
        path = f"{self.__class__.__name__}/{self.model_name}"
        LoggingService().save_figure(
            circuit_fig,
            "circuit_diagram",
            path,
        )

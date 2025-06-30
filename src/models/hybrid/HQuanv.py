"""Hybrid CNN model using a quanvolution layer."""

from typing import Optional, Type

import torch
from torch import flatten, float32
from torch.nn import Linear, LogSoftmax, ReLU

from models.BaseModel import BaseModel
from models.quantum.Quanvolution import Quanvolution
from models.quantum.ansatzs.BaseAnsatz import BaseAnsatz
from models.quantum.ansatzs.RandomStructureAnsatz import RandomStructureAnsatz

class HQuanv(BaseModel):
    """Hybrid Quanvolutional Neural Network."""

    def __init__(
        self,
        in_channels: int,
        classes: int,
        img_size: int = 28,
        patch_size: int = 2,
        stride: Optional[int] = None,
        n_qdepth: int = 1,
        ansatz_cls: Type[BaseAnsatz] = RandomStructureAnsatz,
        embedding: str = "angle",
        rotation: str = "X",
        fc_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters("in_channels", "classes", "img_size", "patch_size", "stride",
                                  "n_qdepth", "ansatz_cls", "embedding", "rotation", "fc_size")

        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride or patch_size

        # Quantum convolution layer
        self.quanv = Quanvolution(
            patch_size=self.patch_size,
            stride=self.stride,
            n_layers=n_qdepth,
            ansatz_cls=ansatz_cls,
            embedding=embedding,
            rotation=rotation,
        )

        n_patches = (self.img_size - self.patch_size) // self.stride + 1
        self.fc_input_dim = fc_size or self.quanv.n_qubits * n_patches * n_patches

        # Classical fully connected layers
        self.fc1 = Linear(in_features=self.fc_input_dim, out_features=self.fc_input_dim)
        self.relu3 = ReLU()

        self.fc2 = Linear(in_features=self.fc_input_dim, out_features=classes)
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

"""Hybrid parallel CNN combined with a quantum layer."""

import torch
from torch import flatten
from torch.nn import BatchNorm2d, Conv2d, Linear, LogSoftmax, MaxPool2d, ReLU

from models import BaseModel, ModelFactory
from models.quantum import ParallelVQC


@ModelFactory.model("hqnn")
class HQNN(BaseModel):
    """Hybrid Quantum Neural Network (HQNN).

    This model consists of a small convolutional feature extractor followed by a
    fully-connected layer.  The resulting feature vector is then processed by a
    stack of variational quantum circuits (:class:`ParallelVQC`).

    Parameters
    ----------
    input_size: int | tuple[int, int] | tuple[int, int, int]
        Shape of the input. The first element denotes the number of channels.
    output_size: int
        Number of output classes.
    in_qlayer: int
        Size of the feature vector fed into the quantum layer.
    qblock_size: int
        Number of features consumed by each quantum block.
    n_qdepth: int, optional
        Depth (number of layers) of each quantum circuit.
    ansatz: dict | None, optional
        Configuration dictionary for the quantum ansatz.
    conv_channels: tuple[int, int], optional
        Output channels for the two convolutional layers.
    **kwargs: Any
        Forwarded to :class:`BaseModel`.
    """

    def __init__(
        self,
        qblock_size: int,
        in_qlayer: int,
        n_qdepth: int = 3,
        ansatz: dict | None = None,
        conv_channels: tuple[int, int] = (16, 32),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters(
            "qblock_size", "in_qlayer", "n_qdepth", "ansatz", "conv_channels"
        )
        self.in_qlayer = in_qlayer

        c1, c2 = conv_channels

        # Convolutional feature extractor
        self.conv1 = Conv2d(
            in_channels=self.input_size[0],
            out_channels=c1,
            kernel_size=(5, 5),
            padding=2,
        )
        self.batchNorm1 = BatchNorm2d(c1)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = Conv2d(
            in_channels=c1, out_channels=c2, kernel_size=(5, 5), padding=2
        )
        self.batchNorm2 = BatchNorm2d(c2)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Determine the feature size after the convolutional stack
        conv_out_size = self._get_conv_out_size()

        # Classical fully-connected layer before the quantum block
        self.fc1 = Linear(in_features=conv_out_size, out_features=in_qlayer)
        self.relu3 = ReLU()

        # Quantum layer configuration
        self.qlayer1 = ParallelVQC(
            input_size=in_qlayer,
            block_size=qblock_size,
            n_layers=n_qdepth,
            ansatz=ansatz,
        )
        self.qinput_size = self.qlayer1.output_size

        # Initialize our softmax classifier
        self.fc2 = Linear(in_features=self.qinput_size, out_features=self.output_size)
        self.logSoftmax = LogSoftmax(dim=1)

    def _get_conv_out_size(self) -> int:
        """Compute the flattened feature size after the convolutional stack."""
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_size)
            x = self.conv1(dummy)
            x = self.batchNorm1(x)
            x = self.relu1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.batchNorm2(x)
            x = self.relu2(x)
            x = self.maxpool2(x)
            return x.view(1, -1).size(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # ==== Convolutional layer
        x = self.conv1(inputs)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = flatten(x, 1)  # flatten the output from the previous layer and pass it

        # ==== FC layer
        x = self.fc1(x)
        x = self.relu3(x)

        # ==== Quantum layer
        x = self.qlayer1(x)

        # ==== Softmax classifier
        x = self.fc2(x)
        output = self.logSoftmax(x)

        return output  # return the output predictions

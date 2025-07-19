"""Hybrid parallel CNN combined with a quantum layer."""


import torch
from torch import flatten
from torch.nn import BatchNorm2d, Conv2d, Linear, LogSoftmax, MaxPool2d, ReLU

from models.BaseModel import BaseModel
from models.quantum.ParallelVQC import ParallelVQC
from metrics import LoggingService



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
    in_qlayer: int, optional
        Size of the feature vector fed into the quantum layer.
    n_qubits: int, optional
        Number of qubits per quantum circuit.  If ``None`` the value of
        ``in_qlayer`` is used.
    n_qdepth: int, optional
        Depth (number of layers) of each quantum circuit.
    **kwargs: Any
        Forwarded to :class:`BaseModel`.
    """

    def __init__(
        self,
        qblock_size: int,
        n_qdepth: int = 3,
        ansatz: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters("qblock_size", "n_qdepth", "ansatz")

        # Convolutional feature extractor
        self.conv1 = Conv2d(
            in_channels=self.input_size[0],
            out_channels=16,
            kernel_size=(5, 5),
            padding=2,
        )
        self.batchNorm1 = BatchNorm2d(16)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(
            in_channels=16, out_channels=32, kernel_size=(5, 5), padding=2
        )
        self.batchNorm2 = BatchNorm2d(32)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Classical fully-connected layer before the quantum block
        self.fc1 = Linear(in_features=1568, out_features=self.in_qlayer)
        self.relu3 = ReLU()

        # Quantum layer configuration
        self.qlayer1 = ParallelVQC(
            input_size=self.input_size,
            block_size=qblock_size,
            n_layers=n_qdepth,
            ansatz=ansatz,
        )

        # Initialize our softmax classifier
        self.fc2 = Linear(in_features=self.qinput_size, out_features=self.output_size)
        self.logSoftmax = LogSoftmax(dim=1)

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
    
    def on_fit_end(self) -> None:
        circuit_fig = self.qlayer1.blocks[0].ansatz.visualize(True)
        LoggingService().save_figure(
            circuit_fig,
            "circuit_diagram",
            self.log_dir,
        )
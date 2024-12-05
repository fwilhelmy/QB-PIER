# File: VQC.py
import math
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


def create_qnode(n_qubits, circuit, weight_shapes):
    dev = qml.device("lightning.qubit", wires=n_qubits)
    qnode_obj = qml.QNode(circuit, dev, interface="torch", diff_method="parameter-shift"
)
    return qml.qnn.TorchLayer(qnode_obj, weight_shapes)

def strongly_entangling_circuit(n_qubits, n_qdepth, rotation="X"):
    weight_shapes = {"weights": (n_qdepth, n_qubits, 3)}
    
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=rotation)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits), ranges=np.ones(3, dtype=int))
        return [qml.expval(qml.PauliY(wires=i)) for i in range(n_qubits)]
    
    return circuit, weight_shapes

class VQC(nn.Module):
    def __init__(self, size_in, n_qubits, n_qdepth, circuit_type="strongly_entangling"):
        super(VQC, self).__init__()
        self.size_in = size_in
        self.n_qubits = n_qubits
        self.n_qdepth = n_qdepth
        
        # Choose the appropriate circuit type
        if circuit_type == "strongly_entangling":
            self.circuit, self.weight_shapes = strongly_entangling_circuit(n_qubits, n_qdepth)
        else:
            raise ValueError("Invalid circuit type provided.")
        
        self.layers = nn.ModuleList(
            [create_qnode(self.n_qubits, self.circuit, self.weight_shapes) for _ in range(math.ceil(size_in / n_qubits))]
        )

    def forward(self, x):
        # Split the input into the number of qubits
        x_split = torch.split(x, self.n_qubits, dim=1)
        outputs = [layer(inputs) for layer, inputs in zip(self.layers, x_split)]
        # Concatenate the outputs of individual quantum layers
        return torch.cat(outputs, dim=1)

# File: BaseModel.py
from pytorch_lightning import LightningModule
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, MSELoss
import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

def calculate_accuracy(y_hat, y):
    _, predicted = torch.max(y_hat, dim=1)
    correct = (predicted == y).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

SUPPORTED_OPTIMIZERS = {
    "adam": Adam,
    "sgd": SGD
}

SUPPORTED_LOSS_FUNCTIONS = {
    "cross_entropy": CrossEntropyLoss,
    "mse": MSELoss
}

class BaseModel(LightningModule):
    def __init__(
            self,
            model_name: str,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            max_epochs: int = 10,
            optimizer: str = "adam",
            loss_function: str = "cross_entropy",
            seed: int = 42,
            gpus: int = None  # Set to None for CPU training
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        if optimizer.lower() not in SUPPORTED_OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        self.optimizer_class = SUPPORTED_OPTIMIZERS[optimizer.lower()]

        if loss_function.lower() not in SUPPORTED_LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss function: {loss_function}")
        self.loss_function = SUPPORTED_LOSS_FUNCTIONS[loss_function.lower()]()

        self.seed = seed
        self.gpus = gpus

        self._configure_trainer()
        
    def _configure_trainer(self):
        logger = TensorBoardLogger(f"tensorboard", name=self.model_name)
        self.trainer = Trainer(
            max_epochs=self.max_epochs,
            logger=logger,
            default_root_dir="tensorboard"
        )

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        self.log_dict({
            "train_loss": loss, 
            "train_acc": acc
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        self.log_dict({
            "val_loss": loss, 
            "val_acc": acc
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True)
        
        return loss

    def fit(self, train_dataloaders, val_dataloaders):
        self.trainer.fit(self, train_dataloaders, val_dataloaders)

# File: HQNN_Parallel.py
from torch.nn import Conv2d, Linear, MaxPool2d, ReLU, BatchNorm2d, LogSoftmax
from torch import flatten
import torch
from torch.optim import Adam
import time
import torch.nn as nn
from torch.nn import Module
import math

class HQNN_Parallel(BaseModel):
    def __init__(self, in_channels, classes, **kwargs):
        super().__init__(model_name='HQNN_Parallel', **kwargs)

        # Initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(5,5), padding=2)
        self.batchNorm1 = BatchNorm2d(16)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

        # Initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5), padding=2)
        self.batchNorm2 = BatchNorm2d(32)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

        # Initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=1568, out_features=20) 
        self.relu3 = ReLU()

        # Initialize the quantum layer
        self.qlayer1 = VQC(
            size_in=20,
            n_qubits=5,
            n_qdepth=3,
        )

        # Initialize our softmax classifier
        self.fc2 = Linear(in_features=20, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # ==== Convolutional layers
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = flatten(x, 1)  # flatten the output from the previous layer

        # ==== Fully Connected layer
        x = self.fc1(x)
        x = self.relu3(x)

        # ==== Quantum layer
        x = self.qlayer1(x)

        # ==== Softmax classifier
        x = self.fc2(x)
        output = self.logSoftmax(x)

        return output  # return the output predictions

    def fit(self, trainDataLoader, valDataLoader):
        self.trainer.fit(self, trainDataLoader, valDataLoader)

# File: Main.py
from torch.utils.data import DataLoader, TensorDataset
import torch

if __name__ == "__main__":
    # Create dummy datasets
    X_train = torch.randn(1500, 1, 28, 28)
    y_train = torch.randint(0, 10, (1500,))
    X_val = torch.randn(300, 1, 28, 28)
    y_val = torch.randint(0, 10, (300,))

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Initialize and train the model
    model = HQNN_Parallel(in_channels=1, classes=10)
    model.fit(train_loader, val_loader)
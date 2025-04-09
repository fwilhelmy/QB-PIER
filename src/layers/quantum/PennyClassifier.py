import pennylane as qml
from pennylane import numpy as np
import torch
from matplotlib import pyplot as plt
from models.BaseModel import BaseModel
from utils import create_qnode
import math

# Some of the code comes from : https://pennylane.ai/qml/demos/tutorial_adversarial_attacks_QML
class PennyClassifier(BaseModel):
    """
    Class for creating a quantum machine learning (classification) model based on the StronglyEntanglingLayers template.

    Args:
        input_dim: the dimension of the input samples
        output_dim: the dimension of the output, i.e. the numbers of classes
        num_qubits: the number of qubits in the circuit
        num_layers: the number of layers within the StronglyEntanglingLayers template
    """
    def __init__(self, in_dim, in_channels, classes, num_qubits = 10, num_reup = 3, **kwargs):
        # Call the parent constructor
        super().__init__(**kwargs)
        torch.manual_seed(1337)  # fixed seed for reproducibility
        self.num_qubits = num_qubits
        self.output_dim = classes
        self.num_reup = num_reup
        self.num_layers = math.ceil((in_dim*in_dim*80)/self.num_qubits)
        print("num_layers : ", self.num_layers)
        self.weights_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.num_layers, n_wires=self.num_qubits
        )

        def circuit(inputs, weights, bias):
            inputs = torch.reshape(inputs, self.weights_shape)
            qml.StronglyEntanglingLayers(
                weights=weights * inputs + bias, wires=range(self.num_qubits)
            )
            return [qml.expval(qml.PauliZ(i)) for i in range(self.output_dim)]

        param_shapes = {"weights": self.weights_shape, "bias": self.weights_shape}
        init_vals = {
            "weights": 0.1 * torch.rand(self.weights_shape),
            "bias": 0.1 * torch.rand(self.weights_shape),
        }

        # initialize the quantum circuit
        self.qcircuit = create_qnode(num_qubits, circuit, param_shapes, init_vals)

    def forward(self, x):
        inputs_stack = torch.hstack([x] * self.num_reup)
        return self.qcircuit(inputs_stack)
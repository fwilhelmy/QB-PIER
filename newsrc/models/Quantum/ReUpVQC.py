import math
import pennylane as qml
import torch
import torch.nn as nn
from torch import Tensor
from utils import create_qtorchlayer

# Could be interesting to explore the true implementation of MaxWeldinger

class ReUpVQC(nn.Module):
    """
    Data-reuploading VQC.
    The input tensor is reshaped per layer and used to modulate
    StronglyEntanglingLayers weights + bias directly.
    """

    def __init__(
        self,
        n_qubits: int,
        n_reup: int,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_reup = n_reup

        # Compute number of entangling layers
        self.num_layers = math.ceil((n_qubits * n_qubits * 80) / self.n_qubits)

        # Define parameter shapes: one weights & bias tensor per layer
        self.weights_shape = (self.num_layers, self.n_qubits, 3)
        params_shape = {
            "weights": self.weights_shape,
            "bias":    self.weights_shape,
        }
        init_vals = {
            "weights": 0.1 * torch.rand(self.weights_shape),
            "bias": 0.1 * torch.rand(self.weights_shape),
        }

        # Set up device and pickle-safe QNode
        self.qlayer = create_qtorchlayer(self.n_qubits, self._circuit, params_shape, init_vals)

    def _circuit(self, inputs, weights, bias):
        """
        QNode: inputs is a flat vector of length product(self.param_shapes["weights"]);
        reshape it per-layer and use it to modulate the entangling-layer weights.
        """
        # reshape inputs to (num_layers, num_qubits, 3)
        data_tensor = inputs.reshape(self.param_shapes["weights"])

        for layer_idx in range(self.num_layers):
            # directly modulate the entangling weights by data + bias
            modulated = weights[layer_idx] * data_tensor[layer_idx] + bias[layer_idx]
            qml.StronglyEntanglingLayers(
                weights=modulated,
                wires=range(self.n_qubits)
            )

        # measure PauliZ on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch_size, in_dim*in_dim*80)
        Repeat x num_reup times horizontally and pass through the QNode.
        """
        stacked = torch.hstack([x] * self.n_reup)
        return self.qlayer(stacked)

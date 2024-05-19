import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
from Utils import create_qnode

def random_layers_circuit(n_qubits, n_qdepth):
    weight_shapes = {"weights": (n_qdepth, n_qubits)}
    
    def circuit(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.RandomLayers(weights, wires=range(n_qubits), ratio_imprim=0.75)
        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

    return circuit, weight_shapes

def fixed_structure_circuit(n_qubits, n_qdepth):
    weight_shapes = {"weights": (n_qdepth, n_qubits)}
    
    def circuit(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.RX(weights[0, 0], wires=0)
        qml.RX(weights[0, 1], wires=1)
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[0, 2])
        qml.CNOT(wires=[0, 3])
        qml.RY(weights[0, 2], wires=0)
        qml.RY(weights[0, 3], wires=3)
        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

    return circuit, weight_shapes

# Quantum convolutional layer class
class QuanvolutionLayer(nn.Module):
    def __init__(self, kernel_size=(2, 2), stride=2, padding=0, n_qdepth=1, circuit_type="fixed_structure"):
        super(QuanvolutionLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_qdepth = n_qdepth
        self.n_qubits = kernel_size[0] * kernel_size[1]
        
        # Choose the appropriate circuit type
        if circuit_type == "random_layers":
            self.circuit, self.weight_shapes = random_layers_circuit(self.n_qubits, self.n_qdepth)
        elif circuit_type == "fixed_structure":
            self.circuit, self.weight_shapes = fixed_structure_circuit(self.n_qubits, self.n_qdepth)
        else:
            raise ValueError("Invalid circuit type provided.")
        
        self.qnode = create_qnode(self.n_qubits, self.circuit, self.weight_shapes)

    def forward(self, x):
        out_shape = (
            x.shape[0],
            self.n_qubits,
            int(x.shape[2] / self.kernel_size[0]),
            int(x.shape[3] / self.kernel_size[1]),
        )
        out = np.zeros(out_shape)
        
        # Loop over the batch size
        for i in range(x.shape[0]):
            # Loop over the coordinates of the top-left pixel of kernel_size squares
            for j in range(0, x.shape[2], self.kernel_size[0]):
                for k in range(0, x.shape[3], self.kernel_size[1]):
                    # Process a region of the image with the quantum circuit
                    q_results = self.qnode(
                        torch.tensor([
                                x[i, 0, j, k],
                                x[i, 0, j, k + 1],
                                x[i, 0, j + 1, k],
                                x[i, 0, j + 1, k + 1],
                    ]))
                    
                    # Assign expectation values to different channels of the output pixel
                    for c in range(self.n_qubits):
                        out[i, c, j // 2, k // 2] = q_results[c]
        
        return torch.tensor(out)

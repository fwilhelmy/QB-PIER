import math
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from utils import create_qnode

def strongly_entangling_circuit(n_qubits, n_qdepth, rotation="X"):
    weight_shapes = {"weights": (n_qdepth, n_qubits, 3)}
    
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=rotation)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits), ranges=np.ones(3, dtype=int))
        return [qml.expval(qml.PauliY(wires=i)) for i in range(n_qubits)]
    
    return circuit, weight_shapes

def basic_entangling_circuit(n_qubits, n_qdepth, rotation="X"):
    weight_shapes = {"weights": (n_qdepth, n_qubits)}
    
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=rotation)
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    
    return circuit, weight_shapes

def no_entanglement_circuit(n_qubits, n_qdepth):
    weight_shapes = {"weights": (0,)}
    
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    return circuit, weight_shapes

# Quantum layer class
class VQC(nn.Module):
    def __init__(self, size_in, n_qubits, n_qdepth, circuit_type="strongly_entangling"):
        super(VQC, self).__init__()
        self.size_in = size_in
        self.n_qubits = n_qubits
        self.n_qdepth = n_qdepth
        
        # Choose the appropriate circuit type
        if circuit_type == "strongly_entangling":
            self.circuit, self.weight_shapes = strongly_entangling_circuit(n_qubits, n_qdepth)
        elif circuit_type == "basic_entangling":
            self.circuit, self.weight_shapes = basic_entangling_circuit(n_qubits, n_qdepth)
        elif circuit_type == "no_entanglement":
            self.circuit, self.weight_shapes = no_entanglement_circuit(n_qubits, n_qdepth)
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

import math
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from Utils import create_qnode

def reupload_circuit(n_qubits, n_qdepth):
    weight_shapes = {"weights": (n_qdepth, n_qubits, 3)}
    
    def circuit(inputs, weights):
        # TODO
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    
    return circuit, weight_shapes

# Quantum layer class
class ReUploadLayer(nn.Module):
    def __init__(self, size_in, n_qubits, n_qdepth):
        super(ReUploadLayer, self).__init__()
        self.size_in = size_in
        self.n_qubits = n_qubits
        self.n_qdepth = n_qdepth
        
        self.circuit, self.weight_shapes = reupload_circuit(n_qubits, n_qdepth)
        self.qnode = create_qnode(self.n_qubits, self.circuit, self.weight_shapes)

    def forward(self, x):
        # Concatenate the outputs of individual quantum layers
        return torch.cat(self.qnode(x), dim=1)

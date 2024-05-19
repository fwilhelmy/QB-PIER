import math
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

# Constants for device configuration
DEVICE = "lightning.qubit"
INTERFACE = "torch"
DIFF_METHOD = "parameter-shift"

# Helper functions for creating circuits
def create_qnode(n_qubits, circuit):
    dev = qml.device(DEVICE, wires=n_qubits)
    qnode_obj, weight_shapes = qml.QNode(circuit, dev, interface=INTERFACE, diff_method=DIFF_METHOD)
    return qml.qnn.TorchLayer(qnode_obj, weight_shapes)

def strongly_entangling_circuit(n_qubits, n_qdepth):
    weight_shapes = {"weights": (n_qdepth, n_qubits, 3)}
    
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits), ranges=np.ones(3, dtype=int))
        return [qml.expval(qml.PauliY(wires=i)) for i in range(n_qubits)]
    
    return circuit, weight_shapes

def basic_entangling_circuit(n_qubits, n_qdepth):
    weight_shapes = {"weights": (n_qdepth, n_qubits)}
    
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    
    return circuit, weight_shapes

def no_entanglement_circuit(n_qubits, n_qdepth):
    weight_shapes = {"weights": (0,)}
    
    @qml.qnode(qml.device(DEVICE, wires=n_qubits))
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    return circuit, weight_shapes

# Quantum layer class
class VQCLayer(nn.Module):
    def __init__(self, size_in, n_qubits, n_qdepth, circuit_type="strongly_entangling"):
        super(VQCLayer, self).__init__()
        self.size_in = size_in
        self.n_qubits = n_qubits
        self.n_qdepth = n_qdepth
        
        # Choose the appropriate circuit type
        if circuit_type == "strongly_entangling":
            self.circuit = strongly_entangling_circuit(n_qubits, n_qdepth)
        elif circuit_type == "basic_entangling":
            self.circuit = basic_entangling_circuit(n_qubits, n_qdepth)
        elif circuit_type == "no_entanglement":
            self.circuit = no_entanglement_circuit(n_qubits, n_qdepth)
        else:
            raise ValueError("Invalid circuit type provided.")
        
        self.circuit = create_qnode(n_qubits, n_qdepth, self.circuit)
        self.layers = nn.ModuleList(
            [self.circuit for _ in range(math.ceil(size_in / n_qubits))]
        )

    def forward(self, x):
        # Split the input into the number of qubits
        x_split = torch.split(x, self.n_qubits, dim=1)
        outputs = [layer(inputs) for layer, inputs in zip(self.layers, x_split)]
        # Concatenate the outputs of individual quantum layers
        return torch.cat(outputs, dim=1)

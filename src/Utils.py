import pennylane as qml
import torch

def get_quantum_device(n_qubits):
    return qml.device("lightning.qubit", wires=n_qubits)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Some of the code taken from
# https://pennylane.ai/qml/demos/tutorial_quanvolution/

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

def FixedStructure(n_qubits, n_qdepth):
    """
    QNode with a fixed circuit structure

    Returns:
        qml.qnn.TorchLayer: A TorchLayer instance
    """

    dev = qml.device("lightning.qubit", wires=n_qubits)
    weight_shapes = {"weights": (n_qdepth, n_qubits)}

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        # for j in range(n_qubits):
        #    qml.RY(np.pi *inputs[j], wires=j)

        qml.RX(weights[0, 0], wires=0)
        qml.RX(weights[0, 1], wires=1)
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[0, 2])
        qml.CNOT(wires=[0, 3])
        qml.RY(weights[0, 2], wires=0)
        qml.RY(weights[0, 3], wires=3)

        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

    return qml.qnn.TorchLayer(circuit, weight_shapes)
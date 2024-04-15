import pennylane as qml
import matplotlib.pyplot as plt
from pennylane import numpy as np


def StronglyEntangling(n_qubits, n_qdepth):
    """
    QNode with strongly entangling layers

    Args:
        n_qubits (int): number of qubits
        n_qdepth (int): number of layers

    Returns:
        qml.qnn.TorchLayer: A TorchLayer instance
    """
    weight_shapes = {"weights": (n_qdepth, n_qubits, 3)}
    dev = qml.device("lightning.qubit", wires=n_qubits)

    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        qml.StronglyEntanglingLayers(
            weights, wires=range(n_qubits), ranges=np.ones(3, dtype=int)
        )
        return [qml.expval(qml.PauliY(wires=i)) for i in range(n_qubits)]

    qnode_obj = qml.QNode(
        circuit, dev, interface="torch", diff_method="parameter-shift"
    )

    return qml.qnn.TorchLayer(qnode_obj, weight_shapes)


def BasicEntangling(n_qubits, n_qdepth):
    """
    QNode with only basic entangling layers

    Args:
        n_qubits (int): number of qubits
        n_qdepth (int): number of layers

    Returns:
        qml.qnn.TorchLayer: A TorchLayer instance
    """
    weight_shapes = {"weights": (n_qdepth, n_qubits)}
    dev = qml.device(
        "lightning.qubit",
        wires=n_qubits,
    )

    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    qnode_obj = qml.QNode(
        circuit, dev, interface="torch", diff_method="parameter-shift"
    )

    return qml.qnn.TorchLayer(qnode_obj, weight_shapes)


def NoEntanglement(n_qubits, n_qdepth):
    """
    QNode without any entangling layers

    Args:
        n_qubits (int): number of qubits
        n_qdepth (int): number of layers

    Returns:
        qml.qnn.TorchLayer: A TorchLayer instance
    """
    weight_shapes = {"weights": (0)}
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    return qml.qnn.TorchLayer(circuit, weight_shapes)
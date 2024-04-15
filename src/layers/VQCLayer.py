import torch
import torch.nn as nn
import pennylane as qml
import math
import matplotlib.pyplot as plt
import VQCCircuits as circuits

class VQCLayer(nn.Module):
    def __init__(self, size_in, n_qubits, n_qdepth):
        super(VQCLayer, self).__init__()
        self.size_in = size_in
        self.n_qubits = n_qubits
        self.n_qdepth = n_qdepth

        self.layers = []
        for _ in range(math.ceil(size_in / n_qubits)):
            # Generate a quantum node for
            self.layers.append(circuits.StronglyEntangling(n_qubits, n_qdepth))

    def forward(self, x):
        # Split the input into the number of qubits
        x_split = torch.split(x, self.n_qubits, dim=1)
        outputs = []
        for layer, inputs in zip(self.layers, x_split):
            outputs.append(layer(inputs))

        # Concatenate the outputs of individual quantum layers
        return torch.cat(outputs, dim=1)


if __name__ == "__main__":
    """
    Prints the circuit diagram of the VQCLayer when the file is run
    """
    dev = qml.device("lightning.qubit", wires=5)

    @qml.qnode(dev)
    @qml.compile()
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(5), rotation="X")
        qml.StronglyEntanglingLayers(weights, wires=range(5))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(5)]

    qml.draw_mpl(circuit)(
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.randn(3, 5, 3),
    )
    plt.show()

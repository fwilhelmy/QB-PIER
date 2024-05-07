import torch
import torch.nn as nn
import pennylane as qml
import math
import matplotlib.pyplot as plt
import layers.VQCCircuits as circuits

class VQCLayer(nn.Module):
    def __init__(self, size_in, n_qubits, n_qdepth):
        super(VQCLayer, self).__init__()
        self.size_in = size_in
        self.n_qubits = n_qubits
        self.n_qdepth = n_qdepth

        self.circuit = circuits.StronglyEntangling
        self.layers = []
        for _ in range(math.ceil(size_in / n_qubits)):
            # Generate a quantum node for
            self.layers.append(self.circuit(n_qubits, n_qdepth))

    def forward(self, x):
        # Split the input into the number of qubits
        x_split = torch.split(x, self.n_qubits, dim=1)
        outputs = []
        for layer, inputs in zip(self.layers, x_split):
            outputs.append(layer(inputs))

        # Concatenate the outputs of individual quantum layers
        return torch.cat(outputs, dim=1)
    
    def printCircuit(self):
        """
        Prints the circuit diagram of the Quanvolution layer
        """
        qml.draw_mpl(self.circuit)(torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.randn(2, 3))
        plt.show()


if __name__ == "__main__":
    """
    Prints the circuit diagram of the Quanvolution layer when the file is run
    """

    # Create an instance of the Quanvolution layer
    layer = VQCLayer(4, 2, 3)
    layer.printCircuit()

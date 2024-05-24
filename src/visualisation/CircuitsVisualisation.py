import matplotlib.pyplot as plt
import os
import pennylane as qml

def printCircuit(self):
    qml.draw_mpl(self.circuit)(torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.randn(2, 3))
    plt.show()
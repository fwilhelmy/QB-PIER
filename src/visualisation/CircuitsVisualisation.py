import matplotlib.pyplot as plt
import torch
import pennylane as qml
from utils import save_figure

# TODO NOT WORKING
def printCircuit(model_name, circuit):
    qml.draw_mpl(circuit)(torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.randn(2, 3))
    
    # Saving the results
    save_figure(plt, model_name)
    plt.close()
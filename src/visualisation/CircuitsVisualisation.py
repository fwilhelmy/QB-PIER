import matplotlib.pyplot as plt
import torch
import pennylane as qml
from lightning.pytorch.utilities import rank_zero_only
from metrics.LoggingService import LoggingService

# TODO NOT WORKING
@rank_zero_only
def printCircuit(model, circuit):
    qml.draw_mpl(circuit)(torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.randn(2, 3))
    
    # Saving the results
    LoggingService().save_figure(plt.gcf(), "circuit", model)


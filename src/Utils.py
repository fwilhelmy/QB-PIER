import pennylane as qml
import torch
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torchvision.transforms import ToTensor

def get_quantum_device(n_qubits):
    return qml.device("lightning.qubit", wires=n_qubits)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_data():
    # Load the datasets for the experiments
    data = MNIST(root="../data", train=True, download=True, transform=ToTensor())

    """ train_split = 0.4
    val_split = 0.4
    test_split = 0.2
    numTrainSamples = int((len(data)) * train_split)
    numValSamples = int((len(data)) * val_split)
    numTestSamples = int((len(data)) * test_split) """

    numTrainSamples = 128
    numValSamples = 128
    numTestSamples = 128

    trainDataset, valDataset, testDataset, _ = random_split(data, [
        numTrainSamples,
        numValSamples,
        numTestSamples,
        len(data) - numTrainSamples - numValSamples - numTestSamples])
    
    return trainDataset, valDataset, testDataset
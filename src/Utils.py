import pennylane as qml
import torch
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
import Configuration as cfg
from torch.optim import Adam
from torch import nn
import os

# Helper functions for creating circuits
def create_qnode(n_qubits, circuit, weight_shapes):
    dev = qml.device(cfg.DEVICE, wires=n_qubits)
    qnode_obj = qml.QNode(circuit, dev, interface=cfg.INTERFACE, diff_method=cfg.DIFF_METHOD)
    return qml.qnn.TorchLayer(qnode_obj, weight_shapes)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model, model_path):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

def get_optimizer(model, init_lr):
    return Adam(model.parameters(), lr=init_lr)

def get_lossfn():
    return nn.CrossEntropyLoss()
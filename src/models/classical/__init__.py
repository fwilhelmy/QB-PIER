"""Classical neural network architectures."""

from src.models.classical.LeNet import LeNet
from src.models.classical.MLP import MLP
from src.models.classical.MLPMixer import MLPMixer
from src.models.classical.ResNet18 import ResNet18
from src.models.classical.ConvNet import ConvNet

__all__ = [
    "LeNet",
    "MLP",
    "MLPMixer",
    "ResNet18",
    "ConvNet",
]

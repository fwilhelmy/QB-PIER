"""Classical neural network architectures."""

from .LeNet import LeNet
from .MLP import MLP
from .MLPMixer import MLPMixer
from .ResNet18 import ResNet18
from .ConvNet import ConvNet

__all__ = [
    "LeNet",
    "MLP",
    "MLPMixer",
    "ResNet18",
    "ConvNet",
]

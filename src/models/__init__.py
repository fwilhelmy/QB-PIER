"""Collection of classical, hybrid and quantum models."""

from src.models.BaseModel import BaseModel
from src.models.classical import LeNet, MLP, MLPMixer, ResNet18, ConvNet
from src.models.hybrid import HQNN, HQuanv, HQE, DimVQC
from src.models.quantum import (
    VQC,
    ParallelVQC,
    Quanvolution,
    BaseAnsatz,
    BasicEntanglingAnsatz,
    StrongEntanglingAnsatz,
    NoEntanglingAnsatz,
    IdentityAnsatz,
    RandomStructureAnsatz,
    ReUpAnsatzWrapper,
    ResidualAnsatzWrapper,
)

__all__ = [
    "BaseModel",
    "LeNet",
    "MLP",
    "MLPMixer",
    "ResNet18",
    "ConvNet",
    "HQNN",
    "HQuanv",
    "DimVQC",
    "VQC",
    "ParallelVQC",
    "Quanvolution",
    "BaseAnsatz",
    "BasicEntanglingAnsatz",
    "StrongEntanglingAnsatz",
    "NoEntanglingAnsatz",
    "IdentityAnsatz",
    "RandomStructureAnsatz",
    "ReUpAnsatzWrapper",
    "ResidualAnsatzWrapper",
]

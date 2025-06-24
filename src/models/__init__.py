"""Collection of classical, hybrid and quantum models."""

from .BaseModel import BaseModel
from .classical import LeNet, MLP, MLPMixer, ResNet18
from .hybrid import HQNN_Parallel, HQNN_Quanv
from .quantum import (
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
    "HQNN_Parallel",
    "HQNN_Quanv",
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

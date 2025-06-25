"""Collection of classical, hybrid and quantum models."""

from .BaseModel import BaseModel
from .classical import LeNet, MLP, MLPMixer, ResNet18
from .hybrid import HQPNN, HQuanvolution, HQE
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
    "HQPNN",
    "HQuanvolution",
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

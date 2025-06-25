"""Collection of classical, hybrid and quantum models."""

from src.models.BaseModel import BaseModel
from src.models.classical import LeNet, MLP, MLPMixer, ResNet18
from src.models.hybrid import HQPNN, HQuanvolution, HQE
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

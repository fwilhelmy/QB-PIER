"""Dataset management utilities."""

from .DatasetManager import DatasetManager
from .DatasetManagerFactory import DatasetManagerFactory
from .MNISTManager import MNISTManager
from .FMNISTManager import FMNISTManager
from .dimensionality_reduction import (
    PCALayer,
    KernelPCALayer,
    RandomProjectionLayer,
    TruncatedSVDLayer,
    UMAPLayer,
)

__all__ = [
    "DatasetManager",
    "DatasetManagerFactory",
    "MNISTManager",
    "FMNISTManager",
    "PCALayer",
    "KernelPCALayer",
    "RandomProjectionLayer",
    "TruncatedSVDLayer",
    "UMAPLayer",
]

"""Dataset management utilities."""

from src.data.DatasetManager import DatasetManager
from src.data.DatasetManagerFactory import DatasetManagerFactory
from src.data.MNISTManager import MNISTManager
from src.data.FMNISTManager import FMNISTManager
from src.data.dimensionality_reduction import (
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

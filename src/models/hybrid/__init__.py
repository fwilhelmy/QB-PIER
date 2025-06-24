"""Hybrid quantum-classical neural networks."""

from .HQNN_Parallel import HQNN_Parallel
from .HQNN_Quanv import HQNN_Quanv

__all__ = [
    "HQNN_Parallel",
    "HQNN_Quanv",
]

"""Hybrid encoder and classifier layers used by the models package."""

from .BaseLayer import BaseLayer
from .LayerFactory import LayerFactory

from . import classical as _classical
from . import hybrid as _hybrid
from . import quantum as _quantum

from .classical import *  # noqa: F401,F403
from .hybrid import *  # noqa: F401,F403
from .quantum import *  # noqa: F401,F403

ParallelVQC = _hybrid.ParallelVQC
QuantumParallelVQC = _quantum.ParallelVQC
ansatz = _quantum.ansatz
embedding = _quantum.embedding

__all__ = [
    "BaseLayer",
    "LayerFactory",
    *_classical.__all__,
    *_hybrid.__all__,
    *_quantum.__all__,
    "QuantumParallelVQC",
]


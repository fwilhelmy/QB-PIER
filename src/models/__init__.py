"""Collection of classical, hybrid and quantum models."""

from .BaseModel import BaseModel
from . import classical
from . import hybrid
from . import quantum

__all__ = [
    "BaseModel",
    "classical",
    "hybrid",
    "quantum",
]

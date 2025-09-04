"""Registry for quantum circuit ansatz classes.

This module refactors the previous dictionary based factory to leverage the
generic :class:`~HydraFactory.HydraFactory`.  The factory now exposes a class
based registry which can be extended dynamically and used to instantiate
registered :class:`BaseAnsatz` implementations.
"""

from __future__ import annotations

from typing import Callable, Dict, Type

from HydraFactory import HydraFactory
from models.quantum.ansatz import BaseAnsatz


class AnsatzFactory(HydraFactory[BaseAnsatz]):
    """Registry and entry point for all :class:`BaseAnsatz` subclasses."""

    _registry: Dict[str, Type[BaseAnsatz]] = {}

    @classmethod
    def ansatz(cls, name: str) -> Callable[[Type[BaseAnsatz]], Type[BaseAnsatz]]:
        """Decorator to register a :class:`BaseAnsatz` under ``name``."""

        return super().item(name)


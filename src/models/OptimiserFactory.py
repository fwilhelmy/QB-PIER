"""Factory for instantiating optimisers by name."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Type

import torch
from torch.optim import Optimizer

from HydraFactory import HydraFactory


class OptimiserFactory(HydraFactory[Type[Optimizer]]):
    """Registry-backed factory that builds torch optimisers."""

    @classmethod
    def create(
        cls,
        name: str,
        parameters: Iterable[torch.nn.Parameter],
        learning_rate: float,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optimizer:
        """Instantiate the requested optimiser with the supplied parameters.

        Args:
            name: Registered optimiser identifier.
            parameters: Iterable of parameters to optimise.
            learning_rate: Base learning rate to use.
            config: Optional keyword arguments forwarded to the optimiser
                constructor.
        """

        optimizer_cls = cls._get(name)
        kwargs = dict(config or {})
        return optimizer_cls(parameters, lr=learning_rate, **kwargs)


def _register_default_optimisers() -> None:
    """Populate the factory with the standard optimiser set."""

    registry = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }
    for key, optimiser_cls in registry.items():
        OptimiserFactory.register(key, optimiser_cls)


_register_default_optimisers()


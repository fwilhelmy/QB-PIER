"""Factory for creating learning rate schedulers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Type

from torch.optim import Optimizer
from torch.optim import lr_scheduler as lr_schedulers
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from HydraFactory import HydraFactory


class SchedulerFactory(HydraFactory[Type[Any]]):
    """Registry-backed factory that builds scheduler configurations."""

    @classmethod
    def create(
        cls,
        config: Optional[Dict[str, Any]],
        optimizer: Optimizer,
    ) -> Optional[Dict[str, Any]]:
        """Create a scheduler configuration for Lightning.

        Args:
            config: Scheduler configuration mapping. When ``None`` or empty no
                scheduler is created.
            optimizer: Optimiser instance controlled by the scheduler.
        """

        if not config:
            return None

        cfg = deepcopy(config)
        scheduler_name = cfg.pop("name", None)
        if not scheduler_name:
            raise ValueError("Scheduler configuration requires a 'name' field.")

        interval = cfg.pop("interval", "epoch")
        frequency = cfg.pop("frequency", 1)
        monitor = cfg.pop("monitor", None)
        strict = cfg.pop("strict", False)

        scheduler_cls = cls._get(scheduler_name)
        scheduler = scheduler_cls(optimizer, **cfg)

        scheduler_config: Dict[str, Any] = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": frequency,
            "strict": strict,
        }

        if monitor is not None:
            scheduler_config["monitor"] = monitor
        elif isinstance(scheduler, ReduceLROnPlateau):
            scheduler_config["monitor"] = "val/loss"

        return scheduler_config


def _register_default_schedulers() -> None:
    """Register scheduler classes exposed by :mod:`torch.optim.lr_scheduler`."""

    for attr_name in dir(lr_schedulers):
        attr = getattr(lr_schedulers, attr_name)
        if not isinstance(attr, type):
            continue
        if not issubclass(attr, (_LRScheduler, ReduceLROnPlateau)):
            continue
        SchedulerFactory.register(attr_name, attr)


_register_default_schedulers()


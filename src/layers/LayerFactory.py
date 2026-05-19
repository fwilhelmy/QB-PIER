"""Factory for registering and instantiating layers."""

import inspect
from typing import Any, Callable, Type

from torch.nn import Module

from HydraFactory import HydraFactory
from . import BaseLayer

class LayerFactory(HydraFactory[BaseLayer]):
    """Registry helper for :class:`BaseLayer` subclasses."""

    @classmethod
    def layer(cls, name: str) -> Callable[[type[Module]], type[BaseLayer]]:
        """Decorator to register a layer under ``name``.

        If a plain :class:`torch.nn.Module` is provided, it is wrapped in a
        lightweight :class:`BaseLayer` subclass before registration.
        """

        def decorator(obj_cls: Type[Module]) -> Type[BaseLayer]:
            if not issubclass(obj_cls, BaseLayer):
                base_cls = obj_cls

                class _WrappedLayer(BaseLayer):
                    def __init__(self, *args: Any, **kwargs: Any) -> None:
                        super().__init__()
                        self.module = base_cls(*args, **kwargs)

                    def forward(self, *args: Any, **kwargs: Any):  # type: ignore[override]
                        return self.module(*args, **kwargs)

                _WrappedLayer.__name__ = f"Wrapped{base_cls.__name__}"
                _WrappedLayer.__qualname__ = _WrappedLayer.__name__
                obj_cls = _WrappedLayer

            return cls.item(name)(obj_cls)

        return decorator

    @classmethod
    def create_layer(cls, name: str, **kwargs: Any) -> BaseLayer:
        """Instantiate the layer registered under ``name``."""

        layer_cls = cls._get(name)
        init_sig = inspect.signature(layer_cls.__init__)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in init_sig.parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in init_sig.parameters
            }
        return layer_cls(**kwargs)


__all__ = ["LayerFactory"]

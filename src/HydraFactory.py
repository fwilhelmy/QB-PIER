from __future__ import annotations

from typing import Any, Callable, Dict, Generic, List, Type, TypeVar

T = TypeVar("T")


class HydraFactory(Generic[T]):
    """Generic registry and factory for subclasses of a given base class."""

    @classmethod
    def _get_registry(cls) -> Dict[str, Type[T]]:
        if "_registry" not in cls.__dict__:
            setattr(cls, "_registry", {})
        return getattr(cls, "_registry")

    @classmethod
    def register(cls, name: str, obj_cls: Type[T]) -> None:
        """Register ``obj_cls`` under ``name``."""
        key = name.lower()
        registry = cls._get_registry()
        registry[key] = obj_cls

    @classmethod
    def item(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a subclass under ``name``."""

        def decorator(obj_cls: Type[T]) -> Type[T]:
            cls.register(name, obj_cls)
            return obj_cls

        return decorator

    @classmethod
    def available(cls) -> List[str]:
        """Return a sorted list of registered names."""
        return sorted(cls._get_registry().keys())

    @classmethod
    def _get(cls, name: str) -> Type[T]:
        registry = cls._get_registry()
        key = name.lower()
        if key not in registry:
            available = ", ".join(cls.available())
            raise KeyError(
                f"Unknown {cls.__name__[:-7].lower()} '{name}'. Available: {available}"
            )
        return registry[key]

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> T:
        """Instantiate a registered class."""
        obj_cls = cls._get(name)
        return obj_cls(**kwargs)

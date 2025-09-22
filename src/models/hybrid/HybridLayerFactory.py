"""Factory for hybrid encoder and classifier layers."""

from typing import Any, Callable

from HydraFactory import HydraFactory

from .HybridLayer import HybridLayer


class HybridLayerFactory(HydraFactory[HybridLayer]):
    """Registry helper for :class:`HybridLayer` subclasses."""

    _ENCODER_PREFIX = "encoder."
    _CLASSIFIER_PREFIX = "classifier."

    @classmethod
    def encoder(cls, name: str) -> Callable[[type[HybridLayer]], type[HybridLayer]]:
        """Decorator to register an encoder under ``name``."""

        return super().item(f"{cls._ENCODER_PREFIX}{name}")

    @classmethod
    def classifier(cls, name: str) -> Callable[[type[HybridLayer]], type[HybridLayer]]:
        """Decorator to register a classifier under ``name``."""

        return super().item(f"{cls._CLASSIFIER_PREFIX}{name}")

    @classmethod
    def create_encoder(cls, name: str, **kwargs: Any) -> HybridLayer:
        """Instantiate the encoder registered under ``name``."""

        return super().create(f"{cls._ENCODER_PREFIX}{name}", **kwargs)

    @classmethod
    def create_classifier(cls, name: str, **kwargs: Any) -> HybridLayer:
        """Instantiate the classifier registered under ``name``."""

        return super().create(f"{cls._CLASSIFIER_PREFIX}{name}", **kwargs)


__all__ = ["HybridLayerFactory"]

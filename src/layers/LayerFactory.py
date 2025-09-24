"""Factory for hybrid encoder and classifier layers."""

from typing import Any, Callable

from ..HydraFactory import HydraFactory

from .BaseLayer import BaseLayer


class LayerFactory(HydraFactory[BaseLayer]):
    """Registry helper for :class:`BaseLayer` subclasses."""

    _ENCODER_PREFIX = "encoder."
    _CLASSIFIER_PREFIX = "classifier."

    @classmethod
    def encoder(cls, name: str) -> Callable[[type[BaseLayer]], type[BaseLayer]]:
        """Decorator to register an encoder under ``name``."""

        return super().item(f"{cls._ENCODER_PREFIX}{name}")

    @classmethod
    def classifier(cls, name: str) -> Callable[[type[BaseLayer]], type[BaseLayer]]:
        """Decorator to register a classifier under ``name``."""

        return super().item(f"{cls._CLASSIFIER_PREFIX}{name}")

    @classmethod
    def create_encoder(cls, name: str, **kwargs: Any) -> BaseLayer:
        """Instantiate the encoder registered under ``name``."""

        return super().create(f"{cls._ENCODER_PREFIX}{name}", **kwargs)

    @classmethod
    def create_classifier(cls, name: str, **kwargs: Any) -> BaseLayer:
        """Instantiate the classifier registered under ``name``."""

        return super().create(f"{cls._CLASSIFIER_PREFIX}{name}", **kwargs)


__all__ = ["LayerFactory"]

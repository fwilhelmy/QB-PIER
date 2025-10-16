"""Tests for :mod:`layers.LayerFactory`."""

from __future__ import annotations

import pytest

from layers import BaseLayer, LayerFactory


@LayerFactory.encoder("_dummy_encoder")
class _DummyEncoder(BaseLayer):
    def __init__(self, value: int) -> None:
        super().__init__(input_size=(1, 1, 1))
        self.value = value

    def forward(self, x):  # pragma: no cover - not used in tests
        return x


@LayerFactory.classifier("_dummy_classifier")
class _DummyClassifier(BaseLayer):
    def __init__(self, output_size: int) -> None:
        super().__init__(in_features=4, output_size=output_size)

    def forward(self, x):  # pragma: no cover - not used in tests
        return x


def test_create_encoder_returns_registered_instance() -> None:
    encoder = LayerFactory.create_encoder("_dummy_encoder", value=7)
    assert isinstance(encoder, _DummyEncoder)
    assert encoder.value == 7


def test_create_classifier_returns_registered_instance() -> None:
    classifier = LayerFactory.create_classifier("_dummy_classifier", output_size=3)
    assert isinstance(classifier, _DummyClassifier)
    assert classifier.output_size == 3


def test_unknown_registration_raises_key_error() -> None:
    with pytest.raises(KeyError):
        LayerFactory.create_encoder("this_does_not_exist")

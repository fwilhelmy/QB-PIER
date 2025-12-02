"""Tests for :mod:`layers.LayerFactory`."""

from __future__ import annotations

import pytest
import torch.nn as nn

from layers import BaseLayer, LayerFactory


@LayerFactory.layer("_dummy_encoder")
class _DummyEncoder(BaseLayer):
    def __init__(self, value: int) -> None:
        super().__init__(input_size=(1, 1, 1))
        self.value = value

    def forward(self, x):  # pragma: no cover - not used in tests
        return x


@LayerFactory.layer("_dummy_classifier")
class _DummyClassifier(BaseLayer):
    def __init__(self, output_size: int) -> None:
        super().__init__(in_features=4, output_size=output_size)

    def forward(self, x):  # pragma: no cover - not used in tests
        return x


def test_create_layer_returns_registered_encoder_instance() -> None:
    encoder = LayerFactory.create_layer("_dummy_encoder", value=7)
    assert isinstance(encoder, _DummyEncoder)
    assert encoder.value == 7


def test_create_layer_returns_registered_classifier_instance() -> None:
    classifier = LayerFactory.create_layer("_dummy_classifier", output_size=3)
    assert isinstance(classifier, _DummyClassifier)
    assert classifier.output_size == 3


def test_unknown_registration_raises_key_error() -> None:
    with pytest.raises(KeyError):
        LayerFactory.create_layer("this_does_not_exist")


def test_layer_decorator_wraps_plain_nn_module() -> None:
    @LayerFactory.layer("_wrapped_linear")
    class _WrappedLinear(nn.Linear):
        pass

    layer = LayerFactory.create_layer("_wrapped_linear", in_features=2, out_features=1)

    assert isinstance(layer, BaseLayer)
    assert isinstance(layer.module, nn.Linear)
    assert layer.module.weight.shape[1] == 2

"""Basic tests for :class:`models.classical.ConvNet`."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.ModelFactory import ModelFactory
from models.classical.ConvNet import ConvNet


@pytest.fixture
def convnet_model(stub_lightning_components, logging_service_root):
    return ConvNet(
        conv_channels=(8, 16),
        fc_sizes=(32,),
        model_path="convnet",
        input_size=(3, 32, 32),
        output_size=6,
    )


def test_forward_returns_logits_with_expected_shape(convnet_model):
    batch = torch.randn(3, 3, 32, 32)

    outputs = convnet_model(batch)

    assert outputs.shape == (3, 6)


def test_model_factory_creates_convnet(stub_lightning_components, logging_service_root):
    model = ModelFactory.create(
        "convnet",
        configure={"optimizer": "adam", "learning_rate": 1e-3},
        conv_channels=(4,),
        fc_sizes=(16,),
        model_path="factory-convnet",
        input_size=(1, 28, 28),
        output_size=2,
    )

    assert isinstance(model, ConvNet)
    assert model.output_size == 2

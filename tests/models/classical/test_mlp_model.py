"""Basic tests for :class:`models.classical.MLP`."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.ModelFactory import ModelFactory
from models.classical.MLP import MLP


@pytest.fixture
def mlp_model(stub_lightning_components, logging_service_root):
    return MLP(
        hidden_sizes=[64, 32],
        model_path="mlp",
        input_size=(3, 32, 32),
        output_size=5,
    )


def test_forward_returns_logits_with_expected_shape(mlp_model):
    batch = torch.randn(4, 3, 32, 32)

    outputs = mlp_model(batch)

    assert outputs.shape == (4, 5)


def test_model_factory_creates_mlp(stub_lightning_components, logging_service_root):
    model = ModelFactory.create(
        "mlp",
        configure={"optimizer": "adam", "learning_rate": 1e-3},
        hidden_sizes=[32, 16],
        model_path="factory-mlp",
        input_size=(1, 28, 28),
        output_size=3,
    )

    assert isinstance(model, MLP)
    assert model.output_size == 3

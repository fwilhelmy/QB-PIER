"""Basic tests for :class:`models.classical.MLPMixer`."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib.pyplot")
torch = pytest.importorskip("torch")

from models.ModelFactory import ModelFactory
from models.classical.MLPMixer import MLPMixer


@pytest.fixture
def mixer_model(stub_lightning_components, logging_service_root):
    return MLPMixer(
        patch_size=4,
        embed_dim=16,
        num_blocks=2,
        model_path="mlpmixer",
        input_size=(3, 32, 32),
        output_size=7,
    )


def test_forward_returns_logits_with_expected_shape(mixer_model):
    batch = torch.randn(2, 3, 32, 32)

    outputs = mixer_model(batch)

    assert outputs.shape == (2, 7)


def test_model_factory_creates_mlpmixer(stub_lightning_components, logging_service_root):
    model = ModelFactory.create(
        "mlpmixer",
        configure={"optimizer": "sgd", "learning_rate": 1e-2},
        patch_size=4,
        embed_dim=32,
        num_blocks=1,
        model_path="factory-mlpmixer",
        input_size=(1, 28, 28),
        output_size=4,
    )

    assert isinstance(model, MLPMixer)
    assert model.output_size == 4

"""Unit tests for :class:`models.classical.ResNet18`."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from layers.classical.ResBlock import ResBlock
from models.ModelFactory import ModelFactory
from models.classical.ResNet18 import ResNet18


@pytest.fixture
def resnet_model(stub_lightning_components, logging_service_root):
    return ResNet18(
        model_path="resnet",
        input_size=(3, 32, 32),
        output_size=5,
    )


def test_forward_produces_logits_with_expected_shape(resnet_model):
    batch = torch.randn(2, 3, 32, 32)
    outputs = resnet_model(batch)

    assert outputs.shape == (2, 5)


def test_make_layer_builds_residual_block_sequence(resnet_model):
    layer = resnet_model._make_layer(64, 128, stride=2)

    assert isinstance(layer, torch.nn.Sequential)
    blocks = list(layer)
    assert len(blocks) == 2
    assert all(isinstance(block, ResBlock) for block in blocks)


def test_visualize_writes_kernel_grid(resnet_model, tmp_path):
    resnet_model.visualize(logdir=str(tmp_path))

    assert (tmp_path / "conv1.png").exists()


def test_model_factory_creates_and_configures_resnet18(stub_lightning_components, logging_service_root):
    model = ModelFactory.create(
        "resnet18",
        configure={"optimizer": "adam", "max_steps": 1},
        model_path="factory-resnet",
        input_size=(3, 32, 32),
        output_size=3,
    )

    assert isinstance(model, ResNet18)
    assert model.hparams["optimizer"] == "adam"
    assert os.path.isdir(model.log_dir)

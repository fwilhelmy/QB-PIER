"""Tests for the :mod:`models.BaseModel` module."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from models.BaseModel import BaseModel


class _SimpleClassifier(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        in_features = self.input_size[0] * self.input_size[1] * self.input_size[2]
        self.layer = torch.nn.Linear(in_features, self.output_size)

    def forward(self, inputs):
        flat = inputs.view(inputs.size(0), -1)
        return self.layer(flat)


@pytest.fixture
def simple_model(stub_lightning_components, logging_service_root):
    return _SimpleClassifier(
        model_path="simple",
        input_size=(1, 8, 8),
        output_size=3,
    )


def test_configure_training_sets_up_trainer(simple_model):
    config = {
        "optimizer": "adam",
        "max_epochs": 5,
        "learning_rate": 1e-2,
        "accelerator": "cpu",
        "devices": 1,
        "fast_dev_run": True,
    }

    simple_model.configure_training(config)

    assert simple_model.hparams["optimizer"] == "adam"
    assert simple_model.hparams["max_epochs"] == 5
    assert simple_model.hparams["fast_dev_run"] is True
    assert isinstance(simple_model.loss_fn, torch.nn.CrossEntropyLoss)
    assert simple_model.trainer.default_root_dir == simple_model.log_dir
    assert simple_model.trainer.fast_dev_run is True

    simple_model._init_directories()
    assert os.path.isdir(simple_model.log_dir)


@pytest.mark.parametrize(
    "optimizer_name, expected_cls",
    [
        ("adamw", torch.optim.AdamW),
        ("adam", torch.optim.Adam),
        ("sgd", torch.optim.SGD),
    ],
)
def test_configure_optimizers_returns_expected_instance(simple_model, optimizer_name, expected_cls):
    simple_model.hparams["learning_rate"] = 1e-2
    simple_model.hparams["optimizer"] = optimizer_name

    optimiser = simple_model.configure_optimizers()

    assert isinstance(optimiser, expected_cls)


def test_configure_optimizers_with_unknown_name(simple_model):
    simple_model.hparams["learning_rate"] = 1e-2
    simple_model.hparams["optimizer"] = "bogus"

    with pytest.raises(ValueError):
        simple_model.configure_optimizers()


def test_training_step_returns_loss_accuracy_and_weights(simple_model):
    simple_model.configure_training({})
    batch = (
        torch.randn(4, *simple_model.input_size),
        torch.randint(0, simple_model.output_size, (4,)),
    )

    outputs = simple_model.training_step(batch, 0)

    assert set(outputs) == {"loss", "acc", "weights_l2"}
    assert outputs["loss"].requires_grad
    assert outputs["acc"].dim() == 0
    assert outputs["weights_l2"] >= 0


def test_validation_and_test_steps(simple_model):
    simple_model.configure_training({})
    batch = (
        torch.randn(2, *simple_model.input_size),
        torch.randint(0, simple_model.output_size, (2,)),
    )

    val_out = simple_model.validation_step(batch, 0)
    test_out = simple_model.test_step(batch, 0)

    assert set(val_out) == {"loss", "acc"}
    assert set(test_out) == {"loss", "acc"}


def test_on_train_epoch_end_logs_gradients(simple_model):
    class _Recorder:
        def __init__(self):
            self.epochs = []
            self.logs = []

        def log_epoch(self, step):
            self.epochs.append(step)

        def log_dict(self, group, metrics, step=None):
            self.logs.append((group, dict(metrics), step))

        def save_metrics(self):
            pass

    recorder = _Recorder()
    simple_model.metrics = recorder  # replace container with spy

    simple_model.global_step = 7
    simple_model._last_grad_l2 = {"layer.weight": 0.5}
    simple_model.on_train_epoch_end()

    assert recorder.epochs == [7]
    assert recorder.logs == [("grad_l2", {"layer.weight": 0.5}, 7)]

    # Fallback to freshly-computed gradients when cache is empty.
    def _fake_gradient_norms():
        return {"layer.bias": 0.25}

    simple_model._last_grad_l2 = {}
    simple_model._gradient_norms = _fake_gradient_norms
    simple_model.on_train_epoch_end()

    assert recorder.logs[-1] == ("grad_l2", {"layer.bias": 0.25}, 7)


def test_on_after_backward_caches_gradient_norms(simple_model):
    simple_model.configure_training({})
    batch = (
        torch.randn(3, *simple_model.input_size),
        torch.randint(0, simple_model.output_size, (3,)),
    )

    outputs = simple_model.training_step(batch, 0)
    outputs["loss"].backward()

    simple_model.on_after_backward()
    assert simple_model._last_grad_l2


def test_checkpoint_round_trip(simple_model, tmp_path, caplog):
    simple_model.configure_training({})

    checkpoint = {}
    simple_model.on_save_checkpoint(checkpoint)
    assert checkpoint["log_dir"] == simple_model.log_dir

    class _Loader:
        def __init__(self):
            self.paths = []

        def load_metrics(self, path):
            self.paths.append(path)

        def save_metrics(self):
            pass

    loader = _Loader()
    simple_model.metrics = loader

    state = {"hyper_parameters": {"optimizer": "adam", "log_dir": str(tmp_path)}}
    simple_model.on_load_checkpoint(state)

    assert isinstance(simple_model.loss_fn, torch.nn.CrossEntropyLoss)
    assert loader.paths == [str(tmp_path)]

    simple_model.metrics = loader
    with caplog.at_level("ERROR", logger="models.BaseModel"):
        simple_model.on_load_checkpoint({"hyper_parameters": {}})
    assert "No log directory provided in checkpoint." in caplog.text

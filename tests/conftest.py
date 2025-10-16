"""Test configuration for QB-PIER."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

# Ensure the project root and ``src`` directory are on ``sys.path`` so imports work consistently.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _stub_pytorch_lightning() -> None:
    """Provide lightweight stand-ins for PyTorch Lightning if it is unavailable."""

    if "pytorch_lightning" in sys.modules:
        return

    pl_module = types.ModuleType("pytorch_lightning")

    try:
        import torch.nn as _nn  # type: ignore
        _base = _nn.Module
    except Exception:  # pragma: no cover - torch is a hard dependency for real tests
        _base = object

    class _LightningModule(_base):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.hparams = {}
            self.global_step = 0

        def save_hyperparameters(self, *names, **kwargs):
            for name in names:
                self.hparams[name] = kwargs.get(name)

        def log_dict(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class _Trainer:
        def __init__(self, *args, **kwargs):
            self.callbacks = list(kwargs.get("callbacks", []))
            self.default_root_dir = kwargs.get("default_root_dir")
            self.max_steps = kwargs.get("max_steps")
            self.accelerator = kwargs.get("accelerator")
            self.devices = kwargs.get("devices")
            self.logger = kwargs.get("logger")
            self.val_check_interval = kwargs.get("val_check_interval")

        def fit(self, *args, **kwargs):
            return None

        def save_checkpoint(self, path):
            return path

    pl_module.LightningModule = _LightningModule
    pl_module.Trainer = _Trainer

    class _StubTensorBoardLogger:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    loggers_module = types.ModuleType("pytorch_lightning.loggers")
    loggers_module.TensorBoardLogger = _StubTensorBoardLogger

    class _StubModelCheckpoint:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    callbacks_module = types.ModuleType("pytorch_lightning.callbacks")
    callbacks_module.ModelCheckpoint = _StubModelCheckpoint

    callback_module = types.ModuleType("pytorch_lightning.callbacks.callback")
    callback_module.Callback = type("Callback", (), {})

    utilities_module = types.ModuleType("pytorch_lightning.utilities")

    def _identity(fn):
        return fn

    utilities_module.rank_zero_only = _identity

    pl_module.loggers = loggers_module
    pl_module.callbacks = callbacks_module
    pl_module.utilities = utilities_module

    sys.modules.update(
        {
            "pytorch_lightning": pl_module,
            "pytorch_lightning.loggers": loggers_module,
            "pytorch_lightning.callbacks": callbacks_module,
            "pytorch_lightning.callbacks.callback": callback_module,
            "pytorch_lightning.utilities": utilities_module,
        }
    )

    callbacks_module.callback = callback_module


def _stub_lightning_package() -> None:
    """Provide a minimal ``lightning`` namespace for metrics utilities."""

    if "lightning" in sys.modules:
        return

    lightning_module = types.ModuleType("lightning")
    lightning_pytorch = types.ModuleType("lightning.pytorch")
    lightning_utilities = types.ModuleType("lightning.pytorch.utilities")

    def _identity(fn):
        return fn

    lightning_utilities.rank_zero_only = _identity
    lightning_pytorch.utilities = lightning_utilities
    lightning_module.pytorch = lightning_pytorch

    sys.modules.update(
        {
            "lightning": lightning_module,
            "lightning.pytorch": lightning_pytorch,
            "lightning.pytorch.utilities": lightning_utilities,
        }
    )


def _stub_psutil() -> None:
    if "psutil" not in sys.modules:
        sys.modules["psutil"] = types.ModuleType("psutil")


def _stub_matplotlib() -> None:
    if "matplotlib" in sys.modules:
        return
    matplotlib_module = types.ModuleType("matplotlib")
    pyplot_module = types.ModuleType("matplotlib.pyplot")

    class _Figure:
        def __init__(self):
            self.saved = []

        def savefig(self, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(b"")
            self.saved.append(path)

    class _Axis:
        def imshow(self, *args, **kwargs):
            return None

        def set_xticks(self, *args, **kwargs):
            return None

        def set_yticks(self, *args, **kwargs):
            return None

        def axis(self, *args, **kwargs):
            return None

    class _Axes(list):
        def flatten(self):
            return self

    def _subplots(rows, cols, figsize=None):
        fig = _Figure()
        axes = _Axes([_Axis() for _ in range(rows * cols)])
        return fig, axes

    pyplot_module.subplots = _subplots
    pyplot_module.close = lambda *args, **kwargs: None
    pyplot_module.tight_layout = lambda *args, **kwargs: None
    matplotlib_module.pyplot = pyplot_module
    sys.modules.update(
        {
            "matplotlib": matplotlib_module,
            "matplotlib.pyplot": pyplot_module,
        }
    )


_stub_pytorch_lightning()
_stub_lightning_package()
_stub_psutil()
_stub_matplotlib()

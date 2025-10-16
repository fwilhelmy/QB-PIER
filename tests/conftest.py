"""Test configuration for QB-PIER."""

from __future__ import annotations

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

    class _LightningModule:
        def __init__(self, *args, **kwargs):
            self.hparams = {}

        def save_hyperparameters(self, *names, **kwargs):
            for name in names:
                self.hparams[name] = kwargs.get(name)

        def log_dict(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

        def parameters(self):
            return []

    class _Trainer:
        def __init__(self, *args, **kwargs):
            self.callbacks = kwargs.get("callbacks", [])
            self.default_root_dir = kwargs.get("default_root_dir")

        def fit(self, *args, **kwargs):
            return None

        def save_checkpoint(self, path):
            return path

    pl_module.LightningModule = _LightningModule
    pl_module.Trainer = _Trainer

    loggers_module = types.ModuleType("pytorch_lightning.loggers")
    loggers_module.TensorBoardLogger = object

    callbacks_module = types.ModuleType("pytorch_lightning.callbacks")
    callbacks_module.ModelCheckpoint = object

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
    pyplot_module.close = lambda *args, **kwargs: None
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

import json
import sys
import types
from pathlib import Path

# Ensure src directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

if "torch" not in sys.modules:
    torch_stub = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    sys.modules["torch"] = torch_stub

if "pandas" not in sys.modules:
    pandas_stub = types.SimpleNamespace(DataFrame=type("DataFrame", (), {}))
    sys.modules["pandas"] = pandas_stub

if "pytorch_lightning" not in sys.modules:
    pl_module = types.ModuleType("pytorch_lightning")
    pl_module.__path__ = []
    pl_module.Callback = object
    pl_module.LightningModule = object
    pl_module.Trainer = object
    sys.modules["pytorch_lightning"] = pl_module

    pl_loggers = types.ModuleType("pytorch_lightning.loggers")
    pl_loggers.__path__ = []
    sys.modules["pytorch_lightning.loggers"] = pl_loggers

    pl_logger_mod = types.ModuleType("pytorch_lightning.loggers.logger")
    pl_logger_mod.Logger = type("Logger", (), {})
    sys.modules["pytorch_lightning.loggers.logger"] = pl_logger_mod

if "lightning.pytorch.utilities" not in sys.modules:
    lightning_pkg = types.ModuleType("lightning")
    lightning_pkg.__path__ = []
    lightning_pytorch_pkg = types.ModuleType("lightning.pytorch")
    lightning_pytorch_pkg.__path__ = []
    lightning_utils = types.ModuleType("lightning.pytorch.utilities")

    def rank_zero_only(fn):
        return fn

    lightning_utils.rank_zero_only = rank_zero_only

    sys.modules["lightning"] = lightning_pkg
    sys.modules["lightning.pytorch"] = lightning_pytorch_pkg
    sys.modules["lightning.pytorch.utilities"] = lightning_utils

from metrics.ExperimentProfiler import ExperimentProfiler


class DummyModule:
    """Simple stand-in for a LightningModule."""

    def __call__(self, *args, **kwargs):  # pragma: no cover - not used but mimics nn.Module
        return None


def test_training_profiler_writes_training_info(tmp_path):
    log_dir = tmp_path / "run"
    profiler = ExperimentProfiler(str(log_dir), input_shape=(1, 28, 28), epochs=5)
    callback = profiler.to_callback()

    callback.on_fit_start(None, DummyModule())
    callback.on_fit_end(None, DummyModule())

    training_file = log_dir / "training.info"
    assert training_file.exists()
    data = json.loads(training_file.read_text())
    assert "pipeline_version" in data
    assert "duration_seconds" in data


def test_training_profiler_invokes_model_info_hook(tmp_path):
    class ModuleWithSaver:
        def __init__(self, log_dir: Path) -> None:
            self.log_dir = log_dir
            self.called_with: float | None = None

        def save_model_info(self, duration: float | None = None) -> None:
            self.called_with = duration
            (self.log_dir / "model.info").write_text(f"duration={duration}")

    log_dir = tmp_path / "run"
    profiler = ExperimentProfiler(str(log_dir), input_shape=(1, 28, 28), epochs=10)
    module = ModuleWithSaver(log_dir)

    callback = profiler.to_callback()
    callback.on_fit_start(None, module)
    callback.on_fit_end(None, module)

    info_file = log_dir / "model.info"
    assert info_file.exists()
    assert module.called_with is not None


def test_experiment_profiler_context_manager(tmp_path):
    log_dir = tmp_path / "run"
    profiler = ExperimentProfiler(str(log_dir), input_shape=(1, 28, 28), epochs=0)

    extra = {"task": "attacks"}
    with profiler.profile(extra_info=extra):
        pass

    training_file = log_dir / "training.info"
    assert training_file.exists()
    data = json.loads(training_file.read_text())
    assert data["task"] == "attacks"

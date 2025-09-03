import sys
import types
import re
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics.LoggingService import LoggingService


class DummyMetricsContainer:
    def __init__(self, path):
        pass
    def load_metrics(self, path):
        pass


def test_load_experiment_sets_logging_root(tmp_path, monkeypatch):
    LoggingService._instance = None

    # Stub modules required by loading
    monkeypatch.setitem(sys.modules, "metrics", types.SimpleNamespace(MetricsContainer=DummyMetricsContainer, LoggingService=LoggingService))
    monkeypatch.setitem(sys.modules, "tqdm", types.SimpleNamespace(tqdm=lambda x, **k: x))

    models_pkg = types.ModuleType("models")
    models_pkg.__path__ = []  # mark as package
    model_factory_mod = types.ModuleType("ModelFactory")
    setattr(model_factory_mod, "ModelFactory", object)
    monkeypatch.setitem(sys.modules, "models", models_pkg)
    monkeypatch.setitem(sys.modules, "models.ModelFactory", model_factory_mod)

    # Import after stubbing
    from loading import load_experiment

    exp_dir = tmp_path / "Deponia"
    exp_dir.mkdir()
    (exp_dir / ".model").write_text("pkg.Model")

    load_experiment(str(exp_dir), load="metrics")

    log_root = Path(LoggingService().root_dir)
    assert log_root.parent == exp_dir
    assert re.match(r"\d{4}-\d{2}-\d{2}@\d{2}-\d{2}-\d{2}", log_root.name)

    LoggingService._instance = None

import importlib
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_analytics_paths_and_notebook_setup_import() -> None:
    importlib.import_module("analytics.paths")
    importlib.import_module("analytics.notebook_setup")


def test_analytics_package_exports() -> None:
    analytics = importlib.import_module("analytics")
    assert set(analytics.__all__) == {"attacks", "model", "tables", "training", "utils"}


def test_resolve_metrics_duck_typing() -> None:
    from analytics.metrics_io import resolve_metrics

    class FakeMetrics:
        log_dir = "."

        def __getitem__(self, key: str):
            return []

    class FakeModel:
        def __init__(self):
            self.metrics = FakeMetrics()

    m = FakeMetrics()
    assert resolve_metrics(m) is m
    assert resolve_metrics(FakeModel()).log_dir == "."

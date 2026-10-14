"""Smoke-test the retained runtime analytics API."""
import importlib

def test_public_analytics_exports():
    analytics = importlib.import_module("analytics")
    assert set(analytics.__all__) == {"training", "utils", "metrics_io", "paths"}

def test_resolve_metrics_duck_typing():
    from analytics.metrics_io import resolve_metrics
    import pytest

    class Metrics:
        log_dir = "."
        def __getitem__(self, key):
            return []

    class Model:
        metrics = Metrics()

    assert resolve_metrics(Metrics()).log_dir == "."
    assert resolve_metrics(Model()).log_dir == "."
    with pytest.raises(TypeError, match="seed dict"):
        resolve_metrics({0: Metrics()})

import importlib
import sys
import types
from pathlib import Path


def test_crawl_experiments(tmp_path, monkeypatch):
    # Provide minimal stubs for optional dependencies so that ``loading`` can be
    # imported without installing them.
    metrics_stub = types.ModuleType("metrics")
    metrics_stub.MetricsContainer = object
    metrics_stub.LoggingService = object
    metrics_stub.TrainingLogger = object
    monkeypatch.setitem(sys.modules, "metrics", metrics_stub)

    monkeypatch.setitem(sys.modules, "tqdm", types.SimpleNamespace(tqdm=lambda x, **k: x))
    monkeypatch.setitem(sys.modules, "main", types.SimpleNamespace(run_attacks=lambda *a, **k: None))

    visualisation_pkg = types.ModuleType("visualisation")
    monkeypatch.setitem(sys.modules, "visualisation", visualisation_pkg)
    monkeypatch.setitem(
        sys.modules,
        "visualisation.training",
        types.SimpleNamespace(
            plot_training_curves=lambda *a, **k: None,
            plot_models_together=lambda *a, **k: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "visualisation.attacks",
        types.SimpleNamespace(plot_transfer_attack_effects=lambda *a, **k: None),
    )

    # Reload loading after stubbing to pick up the fake modules
    sys.modules.pop("loading", None)
    loading = importlib.import_module("loading")
    crawl_experiments = loading.crawl_experiments

    root = tmp_path / "experiments"
    root.mkdir()
    # experiment 1
    exp1 = root / "exp1"
    exp1.mkdir()
    (exp1 / ".model").write_text("pkg.Model")
    # nested experiment 2
    nested = root / "nested"
    nested.mkdir()
    exp2 = nested / "exp2"
    exp2.mkdir()
    (exp2 / ".model").write_text("pkg.Model")
    # directory without experiment
    (root / "empty").mkdir()

    found = crawl_experiments(str(root))
    assert set(map(Path, found)) == {exp1, exp2}

    sys.modules.pop("loading", None)

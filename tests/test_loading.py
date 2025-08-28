from pathlib import Path
import sys
import types

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# Provide minimal stubs for optional dependencies so that ``loading`` can be
# imported without installing them.
sys.modules.setdefault("lightning", types.SimpleNamespace(LightningModule=object))
sys.modules.setdefault("metrics", types.SimpleNamespace(MetricsContainer=object, LoggingService=object))
sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda x, **k: x))
sys.modules.setdefault("main", types.SimpleNamespace(run_attacks=lambda *a, **k: None))
sys.modules.setdefault("visualisation", types.ModuleType("visualisation"))
sys.modules.setdefault("visualisation.training", types.SimpleNamespace(plot_training_curves=lambda *a, **k: None,
                                                                    plot_models_together=lambda *a, **k: None))
sys.modules.setdefault("visualisation.attacks", types.SimpleNamespace(plot_transfer_attack_effects=lambda *a, **k: None))

from loading import crawl_experiments

def test_crawl_experiments(tmp_path):
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

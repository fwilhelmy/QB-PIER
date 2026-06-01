#!/usr/bin/env python3
"""Generate standardized analytics notebooks under src/analytics/notebooks/."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_ROOT = ROOT / "src" / "analytics" / "notebooks"

KERNEL = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

SETUP_CODE = '''import sys
from pathlib import Path

# Add src/ to sys.path before importing analytics (required for Jupyter kernels).
_project_root = Path.cwd().resolve()
while _project_root != _project_root.parent and not (_project_root / "src" / "analytics").is_dir():
    _project_root = _project_root.parent
if not (_project_root / "src" / "analytics").is_dir():
    raise FileNotFoundError(
        "Could not find the project root (expected src/analytics/). "
        "Set the notebook working directory to the QB-PIER repo root."
    )
_src = _project_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from analytics.notebook_setup import notebook_paths, load_all_metrics

project_root, TRAINING_ROOT, ATTACK_EXPERIMENT_DIR, OUT, _src = notebook_paths("{name}", start=_project_root)
'''

MARKDOWN_TEMPLATE = """# {title}

Thin driver for `{import_path}`. Edit config cells only; plotting logic lives in the library module.
"""


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def write_nb(path: Path, title: str, import_path: str, body_cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": [
            md_cell(MARKDOWN_TEMPLATE.format(title=title, import_path=import_path)),
            code_cell(SETUP_CODE.format(name=path.stem)),
            *body_cells,
        ],
        "metadata": {"kernelspec": KERNEL, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def crawl_cell() -> dict:
    return code_cell(
        '''from pathlib import Path
from loading import crawl_experiments, load_metrics

model_dirs = [Path(p) for p in crawl_experiments(str(TRAINING_ROOT))]
print(f"models found: {len(model_dirs)}")
'''
    )


def load_metrics_cell() -> dict:
    return code_cell(
        '''models = load_all_metrics(model_dirs, load_metrics)
print(f"loaded metrics for {len(models)} models")
'''
    )


NOTEBOOKS: list[tuple[str, str, str, list[dict]]] = [
    (
        "_template.ipynb",
        "Analytics notebook template",
        "analytics.notebook_setup",
        [
            md_cell("## Use cases\n\n- **tables/** — CSV inventory and attack metric extraction\n- **training/** — metrics plots from `logger.pkl`\n- **attacks/** — robustness plots from attack tables or live models\n- **model/** — architecture-specific figures"),
            code_cell("# Add your plot call here\n"),
        ],
    ),
    (
        "tables/inventory.ipynb",
        "Training inventory table",
        "analytics.tables.build_inventory_dataframe",
        [
            code_cell(
                '''from analytics.tables import build_inventory_dataframe

df = build_inventory_dataframe(TRAINING_ROOT, load_mode="best")
out_csv = OUT / "inventory.csv"
df.to_csv(out_csv, index=False)
print("Wrote", out_csv)
'''
            ),
        ],
    ),
    (
        "tables/attack_metrics.ipynb",
        "Attack pair metrics table",
        "analytics.tables.build_pair_metrics",
        [
            code_cell(
                '''from pathlib import Path
import pandas as pd
from analytics.tables import build_pair_metrics

pair_df, per_seed_df = build_pair_metrics(
    attack_dir=ATTACK_EXPERIMENT_DIR,
    output_path=OUT / "pair_metrics.csv",
)
print("Wrote", OUT / "pair_metrics.csv")
'''
            ),
        ],
    ),
    (
        "tables/baseline_summary.ipynb",
        "Baseline training summary",
        "analytics.tables.generate_baseline_training_summary",
        [
            code_cell(
                '''from analytics.tables import generate_baseline_training_summary

model_dirs = [str(p) for p in TRAINING_ROOT.rglob("0") if (p / "metrics" / "logger.pkl").exists()]
generate_baseline_training_summary(models=model_dirs[:10], out_dir=str(OUT))
'''
            ),
        ],
    ),
    (
        "training/generalization_gap.ipynb",
        "Generalization gap",
        "analytics.training.generalization_gap",
        [crawl_cell(), load_metrics_cell(), code_cell("from analytics.training import generalization_gap\n\ngeneralization_gap(models=models, out_dir=str(OUT))\n")],
    ),
    (
        "training/single_curves.ipynb",
        "Single model training curves",
        "analytics.training.plot_training_curves",
        [
            crawl_cell(),
            load_metrics_cell(),
            code_cell(
                '''from analytics.training import plot_training_curves

for rel_path, seed_dict in list(models.items())[:3]:
    name = Path(rel_path).name
    plot_training_curves(name, seed_dict, fig_dir=str(OUT / name))
'''
            ),
        ],
    ),
    (
        "training/grouped_curves.ipynb",
        "Grouped training curves",
        "analytics.training.plot_models_together",
        [
            crawl_cell(),
            load_metrics_cell(),
            code_cell(
                '''from analytics.training import plot_models_together

subset = {Path(k).name: v for k, v in list(models.items())[:5]}
plot_models_together(subset, fig_dir=str(OUT))
'''
            ),
        ],
    ),
    (
        "training/gradient_norms.ipynb",
        "Gradient norms",
        "analytics.training.gradient_norms",
        [crawl_cell(), load_metrics_cell(), code_cell("from analytics.training import gradient_norms\n\ngradient_norms(models=models, out_dir=str(OUT))\n")],
    ),
    (
        "training/gradient_flow.ipynb",
        "Gradient flow",
        "analytics.training.plot_gradient_flow",
        [
            crawl_cell(),
            load_metrics_cell(),
            code_cell(
                '''from analytics.training import try_plot_gradient_flow

for rel_path, seed_dict in list(models.items())[:3]:
    try_plot_gradient_flow(seed_dict, fig_dir=str(OUT / Path(rel_path).name))
'''
            ),
        ],
    ),
    (
        "training/gradient_map.ipynb",
        "Gradient map",
        "analytics.training.plot_gradient_map",
        [
            crawl_cell(),
            load_metrics_cell(),
            code_cell(
                '''from analytics.training import try_plot_gradient_map

for rel_path, seed_dict in list(models.items())[:3]:
    try_plot_gradient_map(seed_dict, fig_dir=str(OUT / Path(rel_path).name))
'''
            ),
        ],
    ),
    (
        "training/accuracy_vs_time.ipynb",
        "Accuracy vs time",
        "analytics.training.accuracy_vs_time",
        [crawl_cell(), load_metrics_cell(), code_cell("from analytics.training import accuracy_vs_time\n\naccuracy_vs_time(models=models, out_dir=str(OUT))\n")],
    ),
    (
        "training/batch_all_models.ipynb",
        "Batch training figures",
        "analytics.training.batch",
        [
            code_cell(
                '''from analytics.training.batch import (
    build_mirror_structure,
    load_training_data,
    run_training_figures_for_tree,
)

training_dir = str(TRAINING_ROOT)
build_mirror_structure(training_dir, str(OUT))
loaded = load_training_data(training_dir, map_location="cpu")
run_training_figures_for_tree(training_dir, loaded, str(OUT))
'''
            ),
        ],
    ),
    (
        "training/inference.ipynb",
        "Clean inference",
        "analytics.training.inference",
        [
            code_cell(
                '''from analytics.training.inference import _default_args, run_inference

args = _default_args()
args.models_path = str(TRAINING_ROOT)
args.output_dir = str(OUT)
run_inference(args)
'''
            ),
        ],
    ),
    (
        "attacks/from_csv_accuracy.ipynb",
        "Attack accuracy from CSV",
        "analytics.attacks.plot_attack_accuracy_df",
        [
            code_cell(
                '''from pathlib import Path
import pandas as pd
from analytics.attacks import plot_attack_accuracy_df
from loading import load_attack_aggregates

df = load_attack_aggregates(attack_experiment_dir=ATTACK_EXPERIMENT_DIR)
for source in df["source_model"].unique():
    plot_attack_accuracy_df(df, source, root_dir=str(OUT))
'''
            ),
        ],
    ),
    (
        "attacks/transfer_effects.ipynb",
        "Transfer attack effects",
        "analytics.attacks.plot_transfer_attack_effects",
        [
            code_cell(
                '''import pandas as pd
from analytics.attacks import plot_transfer_attack_effects
from loading import load_attack_aggregates

df = load_attack_aggregates(attack_experiment_dir=ATTACK_EXPERIMENT_DIR)
plot_transfer_attack_effects(df, str(OUT))
'''
            ),
        ],
    ),
    (
        "attacks/attack_matrix.ipynb",
        "Attack effectiveness matrix",
        "analytics.attacks.plot_model_attack_effectiveness_matrix",
        [
            code_cell(
                '''import pandas as pd
from analytics.attacks import (
    plot_model_attack_effectiveness_matrix,
    cell_effectiveness_max_steepness,
)
from loading import load_attack_aggregates

df = load_attack_aggregates(attack_experiment_dir=ATTACK_EXPERIMENT_DIR)
plot_model_attack_effectiveness_matrix(
    df, out_dir=str(OUT), cell_fn=cell_effectiveness_max_steepness
)
'''
            ),
        ],
    ),
    (
        "attacks/confusion_matrix.ipynb",
        "Confusion matrix (live model)",
        "analytics.attacks.plot_confusion_matrix",
        [
            code_cell(
                '''# Configure one model directory and seed, then run evaluation.
MODEL_DIR = None  # e.g. TRAINING_ROOT / "Classical" / "lenet"
'''
            ),
        ],
    ),
    (
        "attacks/perturbation_saliency.ipynb",
        "Perturbation and saliency",
        "analytics.attacks.plot_perturbation_and_saliency",
        [code_cell("# Configure model path and epsilon, then call plot_perturbation_and_saliency\n")],
    ),
    (
        "attacks/loss_landscape.ipynb",
        "Loss landscape",
        "analytics.attacks.plot_loss_landscape",
        [code_cell("# Configure model + batch tensors, then call plot_loss_landscape\n")],
    ),
    (
        "attacks/batch_attack_figures.ipynb",
        "Batch attack figures",
        "analytics.attacks.batch",
        [
            code_cell(
                '''from analytics.attacks.batch import run_attack_figures_for_tree

run_attack_figures_for_tree(
    model_logdir=str(TRAINING_ROOT),
    output_dir=str(OUT),
)
'''
            ),
        ],
    ),
    (
        "model/hquanv.ipynb",
        "HQANV visualization",
        "analytics.model.run_hquanv_batch",
        [
            code_cell(
                '''from analytics.model import run_hquanv_batch

run_hquanv_batch(experiment_root=str(TRAINING_ROOT), output_root=str(OUT))
'''
            ),
        ],
    ),
    (
        "model/ansatz_circuits.ipynb",
        "Ansatz circuit diagrams",
        "analytics.model.ansatz",
        [
            code_cell(
                '''from analytics.model.ansatz import _default_args, main

args = _default_args()
args.output_dir = str(OUT)
# Edit depth, n_qubits, etc. on args before calling main()
main()
'''
            ),
        ],
    ),
]


def main() -> None:
    for rel, title, import_path, extra in NOTEBOOKS:
        write_nb(NB_ROOT / rel, title, import_path, extra)
    print(f"Wrote {len(NOTEBOOKS)} notebooks under {NB_ROOT}")


if __name__ == "__main__":
    main()

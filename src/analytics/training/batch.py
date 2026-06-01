"""Batch training figure generation over an experiment tree."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Dict

from tqdm.auto import tqdm

from loading import crawl_experiments, load_metrics
from analytics.paths import find_project_root, training_log_root, output_dir
from analytics.training import (
    accuracy_vs_time,
    generalization_gap,
    gradient_norms,
    plot_models_together,
    plot_training_curves,
    try_plot_gradient_flow,
    try_plot_gradient_map,
)

log = logging.getLogger(__name__)


def mirror_dir(source_root: str, source_path: str, output_root: str | None) -> str:
    if output_root is None:
        return source_path

    rel_path = os.path.relpath(source_path, source_root)
    target_path = os.path.join(output_root, rel_path)
    os.makedirs(target_path, exist_ok=True)
    return target_path


def build_mirror_structure(source_root: str, output_root: str | None) -> None:
    if output_root is None:
        return

    for current_dir, _, _ in tqdm(
        os.walk(source_root),
        desc=f"Mirroring directories from {source_root}",
        unit="dir",
    ):
        mirror_dir(source_root, current_dir, output_root)


def load_training_data(training_dir: str, map_location: str) -> Dict[str, Dict[int, object]]:
    """Load per-seed MetricsContainers for each model directory."""
    _ = map_location  # retained for call-site compatibility; metrics loading is CPU-only.
    loaded_models: Dict[str, Dict[int, object]] = {}
    model_dirs = list(crawl_experiments(training_dir))
    for model_dir in tqdm(model_dirs, desc="Loading metrics", unit="model"):
        rel_model_dir = os.path.relpath(model_dir, training_dir)
        loaded_models[rel_model_dir] = load_metrics(model_dir)
    return loaded_models


def run_training_figures_for_tree(
    training_dir: str,
    loaded_models: Dict[str, Dict[int, object]],
    output_dir: str | None,
) -> None:
    """Generate per-model and grouped training figures under *output_dir*."""
    grouped_models: dict[str, dict[str, dict[int, object]]] = defaultdict(dict)

    for rel_model_dir, seed_dict in tqdm(
        loaded_models.items(),
        desc="Generating per-model figures",
        unit="model",
    ):
        source_model_dir = os.path.join(training_dir, rel_model_dir)
        target_model_dir = mirror_dir(training_dir, source_model_dir, output_dir)

        model_name = os.path.basename(source_model_dir)
        plot_training_curves(model_name, seed_dict, fig_dir=target_model_dir)

        try_plot_gradient_flow(seed_dict, fig_dir=target_model_dir)
        try_plot_gradient_map(seed_dict, fig_dir=target_model_dir)

        rel_parent = os.path.dirname(rel_model_dir)
        grouped_models[rel_parent][model_name] = seed_dict
        accuracy_vs_time(models={model_name: seed_dict}, out_dir=target_model_dir)
        generalization_gap(models={model_name: seed_dict}, out_dir=target_model_dir)
        gradient_norms(models={model_name: seed_dict}, out_dir=target_model_dir)

    for rel_parent_dir, models_in_group in tqdm(
        grouped_models.items(),
        desc="Generating grouped figures",
        unit="group",
    ):
        source_parent_dir = os.path.join(training_dir, rel_parent_dir)
        target_parent_dir = mirror_dir(training_dir, source_parent_dir, output_dir)

        plot_models_together(models_in_group, fig_dir=target_parent_dir)
        accuracy_vs_time(models=models_in_group, out_dir=target_parent_dir)
        generalization_gap(models=models_in_group, out_dir=target_parent_dir)
        gradient_norms(models=models_in_group, out_dir=target_parent_dir)


def main() -> None:
    project_root = find_project_root()
    training_dir = str(training_log_root(project_root))
    output_dir = str(output_dir(project_root, "visualize"))
    map_location = "cpu"

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    build_mirror_structure(training_dir, output_dir)
    loaded_models = load_training_data(training_dir, map_location)
    run_training_figures_for_tree(training_dir, loaded_models, output_dir)


if __name__ == "__main__":
    main()

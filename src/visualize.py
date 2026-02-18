"""Offline visualisation runner for trained experiments."""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from typing import Dict

from tqdm.auto import tqdm

from loading import crawl_experiments, load_model
from visualisation.training import (
    generate_arch_accuracy_vs_time,
    generate_arch_generalization_gap,
    generate_arch_gradient_norms,
    generate_baseline_training_summary,
    generate_gradient_flow_plots,
    plot_models_together,
    plot_training_curves,
    generate_gradient_map_plots
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
    loaded_models: Dict[str, Dict[int, object]] = {}
    model_dirs = list(crawl_experiments(training_dir))
    for model_dir in tqdm(model_dirs, desc="Loading model checkpoints", unit="model"):
        rel_model_dir = os.path.relpath(model_dir, training_dir)
        loaded_models[rel_model_dir] = load_model(model_dir, best=False, map_location=map_location)
    return loaded_models


def generate_training_figures(
    training_dir: str,
    loaded_models: Dict[str, Dict[int, object]],
    output_dir: str | None,
) -> None:
    grouped_models: dict[str, dict[str, dict[int, object]]] = defaultdict(dict)
    grouped_paths: dict[str, list[str]] = defaultdict(list)

    for rel_model_dir, seed_dict in tqdm(
        loaded_models.items(),
        desc="Generating per-model figures",
        unit="model",
    ):
        source_model_dir = os.path.join(training_dir, rel_model_dir)
        target_model_dir = mirror_dir(training_dir, source_model_dir, output_dir)

        model_name = os.path.basename(source_model_dir)
        plot_training_curves(model_name, seed_dict, fig_dir=target_model_dir)

        generate_gradient_flow_plots(seed_dict, fig_dir=target_model_dir)
        generate_gradient_map_plots(seed_dict, fig_dir=target_model_dir)

        rel_parent = os.path.dirname(rel_model_dir)
        grouped_models[rel_parent][model_name] = seed_dict
        grouped_paths[rel_parent].append(source_model_dir)
        generate_arch_accuracy_vs_time(models={model_name: seed_dict}, out_dir=target_model_dir)
        generate_arch_generalization_gap(models=[seed_dict], out_dir=target_model_dir)
        generate_arch_gradient_norms(models={model_name: seed_dict}, out_dir=target_model_dir)
        generate_baseline_training_summary(models=[source_model_dir], out_dir=target_model_dir)

    for rel_parent_dir, models_in_group in tqdm(
        grouped_models.items(),
        desc="Generating grouped figures",
        unit="group",
    ):
        source_parent_dir = os.path.join(training_dir, rel_parent_dir)
        target_parent_dir = mirror_dir(training_dir, source_parent_dir, output_dir)

        plot_models_together(models_in_group, fig_dir=target_parent_dir)
        generate_arch_accuracy_vs_time(models=models_in_group, out_dir=target_parent_dir)
        generate_arch_generalization_gap(models=models_in_group, out_dir=target_parent_dir)
        generate_arch_gradient_norms(models=models_in_group, out_dir=target_parent_dir)
        generate_baseline_training_summary(models=grouped_paths[rel_parent_dir], out_dir=target_parent_dir)

def generate_attack_figures(attacks_dir: str, output_dir: str | None) -> None:
    """TODO: Add attack-figure generation pipeline."""

    # TODO: implement attack figure generation.
    _ = attacks_dir, output_dir

def main() -> None:
    training_dir = "./logs/ByModel"
    attack_dir = None
    output_dir = "./logs/test"
    map_location = "cpu"

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    build_mirror_structure(training_dir, output_dir)
    loaded_models = load_training_data(training_dir, map_location)
    generate_training_figures(training_dir, loaded_models, output_dir)

    if attack_dir is not None:
        build_mirror_structure(attack_dir, output_dir)
        generate_attack_figures(attack_dir, output_dir)


if __name__ == "__main__":
    main()

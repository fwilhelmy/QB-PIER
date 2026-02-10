"""Offline visualisation runner for trained experiments."""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from typing import Dict

from loading import crawl_experiments, load_model
from visualisation.training import (
    generate_gradient_flow_plots,
    plot_models_together,
    plot_training_curves,
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

    for current_dir, _, _ in os.walk(source_root):
        mirror_dir(source_root, current_dir, output_root)


def load_training_data(training_dir: str, map_location: str) -> Dict[str, Dict[int, object]]:
    loaded_models: Dict[str, Dict[int, object]] = {}
    for model_dir in crawl_experiments(training_dir):
        rel_model_dir = os.path.relpath(model_dir, training_dir)
        loaded_models[rel_model_dir] = load_model(model_dir, best=False, map_location=map_location)
    return loaded_models


def generate_training_figures(
    training_dir: str,
    loaded_models: Dict[str, Dict[int, object]],
    output_dir: str | None,
) -> None:
    grouped_models: dict[str, dict[str, dict[int, object]]] = defaultdict(dict)

    for rel_model_dir, seed_dict in loaded_models.items():
        source_model_dir = os.path.join(training_dir, rel_model_dir)
        target_model_dir = mirror_dir(training_dir, source_model_dir, output_dir)

        model_name = os.path.basename(source_model_dir)
        plot_training_curves(model_name, seed_dict, fig_dir=target_model_dir)

        first_seed = min(seed_dict)
        generate_gradient_flow_plots(seed_dict[first_seed], fig_dir=target_model_dir)

        rel_parent = os.path.dirname(rel_model_dir)
        grouped_models[rel_parent][model_name] = seed_dict

    for rel_parent_dir, models_in_group in grouped_models.items():
        source_parent_dir = os.path.join(training_dir, rel_parent_dir)
        target_parent_dir = mirror_dir(training_dir, source_parent_dir, output_dir)
        plot_models_together(models_in_group, fig_dir=target_parent_dir)


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

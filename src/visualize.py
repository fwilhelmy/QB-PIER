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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training and attack figures from saved runs.")
    parser.add_argument(
        "--training-dir",
        required=True,
        help="Root directory containing model folders marked by a .model file.",
    )
    parser.add_argument(
        "--attacks-dir",
        default=None,
        help="Optional attack-results directory. If omitted, attack figures are skipped.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output root. When provided, the directory tree from training/attacks inputs is mirrored here.",
    )
    parser.add_argument(
        "--map-location",
        default="cpu",
        help="Device mapping passed to checkpoint loading.",
    )
    return parser.parse_args()


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
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    build_mirror_structure(args.training_dir, args.output_dir)
    loaded_models = load_training_data(args.training_dir, args.map_location)
    generate_training_figures(args.training_dir, loaded_models, args.output_dir)

    if args.attacks_dir:
        build_mirror_structure(args.attacks_dir, args.output_dir)
        generate_attack_figures(args.attacks_dir, args.output_dir)


if __name__ == "__main__":
    main()

"""Refresh revised interface summaries, figures, metadata, and report from native traces."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "analytics" / "attacks" / "mechanistic_propagation_revision.py"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_propagation_revised"


def load_analysis_module():
    specification = importlib.util.spec_from_file_location("mechanistic_propagation_revision", MODULE_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def main() -> None:
    module = load_analysis_module()
    trace = pd.read_csv(OUTPUT_DIR / "sample_level_layerwise_traces.csv")
    layerwise = pd.read_csv(OUTPUT_DIR / "layerwise_summary.csv")
    residual = pd.read_csv(OUTPUT_DIR / "residual_branch_summary.csv")
    interface = pd.read_csv(OUTPUT_DIR / "interface_transition_summary.csv")
    figures = module.generate_revised_figures(trace, OUTPUT_DIR)
    report = module.build_revised_report(trace, layerwise, interface, residual, OUTPUT_DIR)
    print(f"interface_summary_used: {OUTPUT_DIR / 'interface_transition_summary.csv'}")
    for figure in figures:
        print(f"figure: {figure}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()

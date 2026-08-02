"""Run native checkpoint validation, cohort construction, FGSM, and activation tracing."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


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
    outputs = module.run_native_trace_stage(OUTPUT_DIR)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

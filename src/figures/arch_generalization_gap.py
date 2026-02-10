"""Plot per-model generalization gap curves from TensorBoard scalars."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

LOGGER = logging.getLogger(__name__)

REQUIRED_TAGS = {
    "acc": ("train/acc", "val/acc"),
    "loss": ("train/loss", "val/loss"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/val generalization gap per model from TensorBoard logs.",
    )
    parser.add_argument(
        "--models",
        action="append",
        default=[],
        help=(
            "Model run directory or model id. Repeat for multiple models. "
            "If a value points to a text file, each non-empty/non-comment line is read as a model entry."
        ),
    )
    parser.add_argument(
        "--gap_type",
        choices=("loss", "acc"),
        default="loss",
        help=(
            "Gap definition: 'loss' => val/loss - train/loss (default), "
            "'acc' => train/acc - val/acc."
        ),
    )
    parser.add_argument(
        "--metrics_source",
        choices=("tensorboard",),
        default="tensorboard",
        help="Metrics backend. Currently only tensorboard is implemented.",
    )
    parser.add_argument("--out_dir", default="outputs/figures", help="Output directory.")
    parser.add_argument("--format", choices=("pdf", "png"), default="pdf", help="Figure format.")
    parser.add_argument("--tb_size_guidance", type=int, default=0, help="TensorBoard scalar size guidance.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    return parser.parse_args()


def _read_model_entries(models: Sequence[str]) -> List[str]:
    entries: List[str] = []
    for model in models:
        model_path = Path(model)
        if model_path.is_file() and model_path.suffix.lower() in {".txt", ".list"}:
            for line in model_path.read_text(encoding="utf-8").splitlines():
                value = line.strip()
                if value and not value.startswith("#"):
                    entries.append(value)
        else:
            entries.append(model)
    return entries


def _candidate_roots() -> Iterable[Path]:
    cwd = Path.cwd()
    yield cwd
    logs = cwd / "logs"
    if logs.exists():
        yield logs


def _resolve_model_path(entry: str) -> Path | None:
    p = Path(entry)
    if p.exists():
        return p.resolve()

    # Resolve model IDs by searching folders containing TensorBoard event files.
    for root in _candidate_roots():
        for event_file in root.rglob("events.out.tfevents.*"):
            run_dir = event_file.parent
            run_str = str(run_dir)
            if run_dir.name == entry or run_str.endswith(entry):
                return run_dir.resolve()
    return None


def _load_scalar_series(
    model_dir: Path,
    tag: str,
    size_guidance: int,
) -> Tuple[np.ndarray, np.ndarray] | None:
    accumulator = event_accumulator.EventAccumulator(
        str(model_dir),
        size_guidance={event_accumulator.SCALARS: size_guidance},
    )
    try:
        accumulator.Reload()
    except Exception as exc:  # pragma: no cover - external TB parsing errors.
        LOGGER.warning("Failed to read TensorBoard events in %s: %s", model_dir, exc)
        return None

    available = accumulator.Tags().get("scalars", [])
    if tag not in available:
        return None

    events = accumulator.Scalars(tag)
    steps = np.asarray([event.step for event in events], dtype=np.int64)
    values = np.asarray([event.value for event in events], dtype=np.float64)
    return steps, values


def _align_steps(
    lhs_steps: np.ndarray,
    lhs_values: np.ndarray,
    rhs_steps: np.ndarray,
    rhs_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    common_steps = np.intersect1d(lhs_steps, rhs_steps)
    if common_steps.size == 0:
        return common_steps, np.array([]), np.array([])

    lhs_idx = np.searchsorted(lhs_steps, common_steps)
    rhs_idx = np.searchsorted(rhs_steps, common_steps)
    return common_steps, lhs_values[lhs_idx], rhs_values[rhs_idx]


def _derive_label(model_dir: Path) -> str:
    basename = model_dir.name
    parent = model_dir.parent.name
    if basename.lower().startswith("seed") and model_dir.parent.parent != model_dir.parent:
        grandparent = model_dir.parent.parent.name
        return f"{grandparent}/{basename}"
    return f"{parent}/{basename}"


def _build_gap_series(
    model_dir: Path,
    gap_type: str,
    size_guidance: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]] | None:
    train_tag, val_tag = REQUIRED_TAGS[gap_type]

    train_data = _load_scalar_series(model_dir, train_tag, size_guidance)
    val_data = _load_scalar_series(model_dir, val_tag, size_guidance)
    if train_data is None or val_data is None:
        LOGGER.warning(
            "Skipping %s because required tags are missing (needed: %s, %s)",
            model_dir,
            train_tag,
            val_tag,
        )
        return None

    train_steps, train_values = train_data
    val_steps, val_values = val_data

    steps, train_aligned, val_aligned = _align_steps(train_steps, train_values, val_steps, val_values)
    if steps.size == 0:
        LOGGER.warning("Skipping %s because train/val steps do not overlap.", model_dir)
        return None

    if gap_type == "loss":
        gap = val_aligned - train_aligned
    else:
        gap = train_aligned - val_aligned

    return steps, gap, {"train": train_tag, "val": val_tag}


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    raw_entries = _read_model_entries(args.models)
    if not raw_entries:
        raise ValueError("No models were provided. Use --models at least once.")

    resolved_inputs: List[Dict[str, str]] = []
    plotted = 0

    plt.figure(figsize=(10, 6))

    for entry in raw_entries:
        resolved = _resolve_model_path(entry)
        if resolved is None:
            LOGGER.warning("Skipping '%s': could not resolve to a run directory.", entry)
            continue

        series = _build_gap_series(resolved, args.gap_type, args.tb_size_guidance)
        if series is None:
            continue

        steps, gap, tags = series
        label = _derive_label(resolved)
        plt.plot(steps, gap, label=label, linewidth=1.8)
        plotted += 1

        resolved_inputs.append(
            {
                "input": entry,
                "resolved_path": str(resolved),
                "label": label,
                "resolved_scalar_tags": tags,
                "points": int(len(steps)),
            }
        )

    if plotted == 0:
        raise RuntimeError("No model could be plotted. Check paths and scalar tags.")

    ylabel = "val/loss - train/loss" if args.gap_type == "loss" else "train/acc - val/acc"
    plt.title(f"Generalization gap across epochs ({args.gap_type})")
    plt.xlabel("Epoch step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_path = out_dir / f"arch_generalization_gap.{args.format}"
    sidecar_path = out_dir / "arch_generalization_gap.json"

    plt.savefig(fig_path)
    plt.close()

    sidecar = {
        "metrics_source": args.metrics_source,
        "gap_type": args.gap_type,
        "raw_inputs": raw_entries,
        "resolved_models": resolved_inputs,
        "output_figure": str(fig_path),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    LOGGER.info("Saved figure: %s", fig_path)
    LOGGER.info("Saved sidecar: %s", sidecar_path)


if __name__ == "__main__":
    main()

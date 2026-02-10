"""Plot accuracy versus wall-clock time for user-provided model run directories."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SUPPORTED_FORMATS = ("pdf", "png")
X_UNITS_TO_SECONDS = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}


@dataclass
class ScalarSeries:
    """Container for scalar points aligned to elapsed wall-clock time."""

    elapsed_seconds: list[float]
    values: list[float]


@dataclass
class RunSeries:
    """Container for plot series associated to one model run."""

    label: str
    val_acc: ScalarSeries
    train_acc: ScalarSeries | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot validation accuracy versus elapsed wall-clock time for model runs.",
    )
    parser.add_argument(
        "--models",
        action="append",
        required=True,
        help="Model run directory (repeat for each run).",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/figures",
        help="Directory where figure and JSON sidecar are written.",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=SUPPORTED_FORMATS,
        help="Output figure format.",
    )
    parser.add_argument(
        "--include_train",
        action="store_true",
        help="Include train/acc in addition to val/acc for each model.",
    )
    parser.add_argument(
        "--x_units",
        default="minutes",
        choices=tuple(X_UNITS_TO_SECONDS.keys()),
        help="Units for elapsed wall-clock time on the x-axis.",
    )
    return parser.parse_args()


def _default_label(run_dir: Path) -> str:
    parent = run_dir.parent.name
    return f"{parent}/{run_dir.name}" if parent else run_dir.name


def _find_tensorboard_dir(run_dir: Path) -> Path:
    candidates = [
        run_dir / "tensorboard",
        run_dir / "seed_0" / "tensorboard",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    nested = sorted(p for p in run_dir.glob("*/tensorboard") if p.is_dir())
    if nested:
        return nested[0]
    raise FileNotFoundError(f"No tensorboard directory found under: {run_dir}")


def _extract_scalar_series(tb_dir: Path, scalar_tag: str) -> ScalarSeries:
    accumulator = EventAccumulator(str(tb_dir))
    accumulator.Reload()

    if scalar_tag not in accumulator.Tags().get("scalars", []):
        raise ValueError(f"Missing scalar tag '{scalar_tag}' in TensorBoard logs at {tb_dir}")

    events = accumulator.Scalars(scalar_tag)
    if not events:
        raise ValueError(f"Scalar tag '{scalar_tag}' has no events in TensorBoard logs at {tb_dir}")

    first_wall_time = events[0].wall_time
    elapsed_seconds = [event.wall_time - first_wall_time for event in events]
    values = [event.value for event in events]
    return ScalarSeries(elapsed_seconds=elapsed_seconds, values=values)


def _collect_runs(models: Iterable[str], include_train: bool) -> list[RunSeries]:
    runs: list[RunSeries] = []
    for model_str in models:
        run_dir = Path(model_str).expanduser().resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        tb_dir = _find_tensorboard_dir(run_dir)
        val_acc = _extract_scalar_series(tb_dir=tb_dir, scalar_tag="val/acc")
        train_acc = (
            _extract_scalar_series(tb_dir=tb_dir, scalar_tag="train/acc") if include_train else None
        )
        runs.append(RunSeries(label=_default_label(run_dir), val_acc=val_acc, train_acc=train_acc))
    return runs


def _plot_runs(runs: list[RunSeries], x_units: str, include_train: bool, output_path: Path) -> None:
    unit_seconds = X_UNITS_TO_SECONDS[x_units]

    fig, ax = plt.subplots(figsize=(10, 6))
    for run in runs:
        val_x = [value / unit_seconds for value in run.val_acc.elapsed_seconds]
        ax.plot(val_x, run.val_acc.values, label=f"{run.label} val/acc", linewidth=2)

        if include_train and run.train_acc is not None:
            train_x = [value / unit_seconds for value in run.train_acc.elapsed_seconds]
            ax.plot(
                train_x,
                run.train_acc.values,
                label=f"{run.label} train/acc",
                linestyle="--",
                alpha=0.8,
            )

    ax.set_xlabel(f"Elapsed time ({x_units})")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs wall-clock time")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_sidecar(
    output_path: Path,
    model_inputs: list[str],
    resolved_model_paths: list[str],
    include_train: bool,
    x_units: str,
) -> Path:
    payload = {
        "models": model_inputs,
        "resolved_model_paths": resolved_model_paths,
        "include_train": include_train,
        "x_units": x_units,
        "figure_path": str(output_path),
    }
    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sidecar_path


def generate_arch_accuracy_vs_time(
    models: Iterable[str | Path],
    out_dir: str | Path,
    output_format: str = "pdf",
    include_train: bool = False,
    x_units: str = "minutes",
) -> tuple[Path, Path]:
    resolved_models = [str(Path(model).expanduser().resolve()) for model in models]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _collect_runs(models=resolved_models, include_train=include_train)
    figure_path = out_dir / f"arch_accuracy_vs_time.{output_format}"
    _plot_runs(runs=runs, x_units=x_units, include_train=include_train, output_path=figure_path)
    sidecar_path = _write_sidecar(
        output_path=figure_path,
        model_inputs=resolved_models,
        resolved_model_paths=resolved_models,
        include_train=include_train,
        x_units=x_units,
    )
    return figure_path, sidecar_path


def main() -> None:
    args = _parse_args()
    generate_arch_accuracy_vs_time(
        models=args.models,
        out_dir=args.out_dir,
        output_format=args.format,
        include_train=args.include_train,
        x_units=args.x_units,
    )


if __name__ == "__main__":
    main()

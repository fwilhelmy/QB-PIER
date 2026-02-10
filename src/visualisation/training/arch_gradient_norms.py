"""Plot gradient-norm evolution (global/quantum/classical) for provided model runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class ScalarSeries:
    """Scalar points aligned by TensorBoard step."""

    steps: np.ndarray
    values: np.ndarray


def _as_epoch_axis(steps: np.ndarray) -> np.ndarray:
    """Map monotonically increasing steps to 1-based epoch indices."""

    if steps.size == 0:
        return steps
    unique_steps = np.unique(steps)
    mapping = {step: idx + 1 for idx, step in enumerate(sorted(unique_steps.tolist()))}
    return np.array([mapping[s] for s in steps], dtype=float)


def _read_tensorboard_scalars(run_dir: Path) -> dict[str, ScalarSeries]:
    tb_dir = run_dir / "tensorboard"
    if not tb_dir.exists():
        return {}

    event_files = sorted(tb_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return {}

    accumulator = EventAccumulator(str(tb_dir))
    accumulator.Reload()

    all_scalars: dict[str, ScalarSeries] = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        if not events:
            continue
        steps = np.array([int(evt.step) for evt in events], dtype=int)
        values = np.array([float(evt.value) for evt in events], dtype=float)
        all_scalars[tag] = ScalarSeries(steps=steps, values=values)
    return all_scalars


def _rms(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(math.sqrt(np.mean(np.square(arr))))


def _aggregate_by_step_rms(series_by_tag: dict[str, ScalarSeries], tags: list[str]) -> ScalarSeries | None:
    if not tags:
        return None

    values_by_step: dict[int, list[float]] = {}
    for tag in tags:
        series = series_by_tag[tag]
        for step, value in zip(series.steps.tolist(), series.values.tolist()):
            values_by_step.setdefault(step, []).append(value)

    if not values_by_step:
        return None

    steps = np.array(sorted(values_by_step.keys()), dtype=int)
    values = np.array([_rms(values_by_step[int(step)]) for step in steps], dtype=float)
    return ScalarSeries(steps=steps, values=values)


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    text_lower = text.lower()
    return any(needle in text_lower for needle in needles)


def _pick_direct_tag(series_by_tag: dict[str, ScalarSeries], include: Iterable[str], exclude: Iterable[str] = ()) -> str | None:
    include_lower = [item.lower() for item in include]
    exclude_lower = [item.lower() for item in exclude]

    candidates = []
    for tag in series_by_tag:
        tag_lower = tag.lower()
        if all(token in tag_lower for token in include_lower) and not _contains_any(tag_lower, exclude_lower):
            candidates.append(tag)

    if not candidates:
        return None

    candidates.sort(key=lambda t: (len(t), t))
    return candidates[0]


def _find_per_parameter_tags(series_by_tag: dict[str, ScalarSeries], qname_pattern: str) -> tuple[list[str], list[str]]:
    quantum_tags: list[str] = []
    classical_tags: list[str] = []
    q_re = re.compile(re.escape(qname_pattern), flags=re.IGNORECASE)

    for tag in series_by_tag:
        lower = tag.lower()
        if not lower.startswith("grad_l2/"):
            continue
        param_name = tag.split("/", 1)[1]
        if q_re.search(param_name):
            quantum_tags.append(tag)
        else:
            classical_tags.append(tag)

    return sorted(quantum_tags), sorted(classical_tags)


def _resolve_gradient_series(
    series_by_tag: dict[str, ScalarSeries],
    qname_pattern: str,
) -> tuple[ScalarSeries | None, ScalarSeries | None, ScalarSeries | None, dict[str, object]]:
    # Prefer explicitly logged aggregate series if they exist.
    direct_global_tag = _pick_direct_tag(series_by_tag, include=["grad", "norm"], exclude=["qlayer", "quant", "classical"])
    direct_quantum_tag = _pick_direct_tag(series_by_tag, include=["grad", "norm", "quant"])
    direct_classical_tag = _pick_direct_tag(series_by_tag, include=["grad", "norm", "classical"])

    global_series = series_by_tag.get(direct_global_tag) if direct_global_tag else None
    quantum_series = series_by_tag.get(direct_quantum_tag) if direct_quantum_tag else None
    classical_series = series_by_tag.get(direct_classical_tag) if direct_classical_tag else None

    q_param_tags, c_param_tags = _find_per_parameter_tags(series_by_tag, qname_pattern=qname_pattern)

    # Fallback aggregation from per-parameter gradient norms (RMS across params per step).
    if quantum_series is None and q_param_tags:
        quantum_series = _aggregate_by_step_rms(series_by_tag, q_param_tags)
    if classical_series is None and c_param_tags:
        classical_series = _aggregate_by_step_rms(series_by_tag, c_param_tags)

    if global_series is None:
        all_param_tags = sorted(q_param_tags + c_param_tags)
        if all_param_tags:
            global_series = _aggregate_by_step_rms(series_by_tag, all_param_tags)

    metadata = {
        "direct_tags": {
            "global": direct_global_tag,
            "quantum": direct_quantum_tag,
            "classical": direct_classical_tag,
        },
        "aggregated_from_param_tags": {
            "quantum": q_param_tags if direct_quantum_tag is None else [],
            "classical": c_param_tags if direct_classical_tag is None else [],
            "global": sorted(q_param_tags + c_param_tags) if direct_global_tag is None else [],
        },
        "aggregation": "rms",
    }

    return global_series, quantum_series, classical_series, metadata


def _model_label(run_dir: Path) -> str:
    return run_dir.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        action="append",
        required=True,
        help="Model run directory (repeatable). Expected to contain tensorboard events.",
    )
    parser.add_argument("--qname_pattern", default="qlayer", help="Substring used to identify quantum parameter names.")
    parser.add_argument("--out_dir", default="outputs/figures", help="Output directory for figure + JSON sidecar.")
    parser.add_argument("--format", choices=["png", "pdf"], default="pdf", help="Figure format.")
    return parser



def generate_arch_gradient_norms(
    models: Iterable[str | Path],
    out_dir: str | Path,
    qname_pattern: str = "qlayer",
    output_format: str = "pdf",
) -> tuple[Path, Path]:
    run_dirs = [Path(path).expanduser().resolve() for path in models]
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sidecar: dict[str, object] = {
        "qname_pattern": qname_pattern,
        "models": [],
        "aggregation": "rms",
    }

    plotted_any = False
    for run_dir in run_dirs:
        scalars = _read_tensorboard_scalars(run_dir)
        global_series, quantum_series, classical_series, meta = _resolve_gradient_series(
            scalars, qname_pattern=qname_pattern
        )

        model_summary = {
            "run_dir": str(run_dir),
            "resolved": meta,
            "num_points": {
                "global": int(global_series.values.size) if global_series is not None else 0,
                "quantum": int(quantum_series.values.size) if quantum_series is not None else 0,
                "classical": int(classical_series.values.size) if classical_series is not None else 0,
            },
        }
        sidecar["models"].append(model_summary)

        label = _model_label(run_dir)
        print(
            f"[{label}] tags: {meta['direct_tags']} | "
            f"points(global={model_summary['num_points']['global']}, "
            f"quantum={model_summary['num_points']['quantum']}, "
            f"classical={model_summary['num_points']['classical']})"
        )

        if global_series is not None:
            ax.plot(
                _as_epoch_axis(global_series.steps),
                global_series.values,
                linewidth=2.0,
                label=f"{label} — global",
            )
            plotted_any = True

        if quantum_series is not None:
            ax.plot(
                _as_epoch_axis(quantum_series.steps),
                quantum_series.values,
                linestyle="--",
                linewidth=1.7,
                label=f"{label} — quantum",
            )
            plotted_any = True

        if classical_series is not None:
            ax.plot(
                _as_epoch_axis(classical_series.steps),
                classical_series.values,
                linestyle=":",
                linewidth=1.7,
                label=f"{label} — classical",
            )
            plotted_any = True

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient norm")
    ax.set_title("Gradient norm evolution")
    ax.grid(alpha=0.25)
    if plotted_any:
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()

    figure_path = out_dir / f"arch_gradient_norms.{output_format}"
    sidecar_path = out_dir / "arch_gradient_norms.tags.json"
    fig.savefig(figure_path, dpi=220 if output_format == "png" else None)
    plt.close(fig)

    with sidecar_path.open("w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    print(f"Saved figure: {figure_path}")
    print(f"Saved sidecar: {sidecar_path}")
    return figure_path, sidecar_path

def main() -> None:
    args = build_parser().parse_args()
    generate_arch_gradient_norms(
        models=args.models,
        out_dir=args.out_dir,
        qname_pattern=args.qname_pattern,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()

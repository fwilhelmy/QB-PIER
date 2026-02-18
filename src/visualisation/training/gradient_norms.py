"""Plot gradient-norm evolution (global/quantum/classical) from loaded model dictionaries."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

@dataclass
class ScalarSeries:
    """Scalar points aligned by training step."""

    steps: np.ndarray
    values: np.ndarray


def _as_epoch_axis(steps: np.ndarray) -> np.ndarray:
    """Map monotonically increasing steps to 1-based epoch indices."""

    if steps.size == 0:
        return steps
    unique_steps = np.unique(steps)
    mapping = {step: idx + 1 for idx, step in enumerate(sorted(unique_steps.tolist()))}
    return np.array([mapping[s] for s in steps], dtype=float)


def _aggregate_by_step_rms(series_by_metric: Mapping[str, np.ndarray], metric_names: list[str]) -> np.ndarray | None:
    if not metric_names:
        return None

    available = [name for name in metric_names if name in series_by_metric]
    if not available:
        return None
    min_len = min(np.asarray(series_by_metric[name]).size for name in available)
    if min_len == 0:
        return None

    aligned = np.vstack([np.asarray(series_by_metric[name], dtype=float)[:min_len] for name in available])
    return np.sqrt(np.mean(np.square(aligned), axis=0))


def _resolve_gradient_series(
    seed_dict: Mapping[int, object],
    qname_pattern: str,
) -> tuple[ScalarSeries | None, ScalarSeries | None, ScalarSeries | None, dict[str, object]]:
    global_curves: list[np.ndarray] = []
    quantum_curves: list[np.ndarray] = []
    classical_curves: list[np.ndarray] = []
    step_curves: list[np.ndarray] = []
    seed_metadata: list[dict[str, object]] = []

    for seed, model in seed_dict.items():
        grad_metrics = model.metrics["grad_l2"]
        steps = np.asarray(model.metrics["grad_l2/step"])
        if grad_metrics is None or steps is None:
            seed_metadata.append({"seed": seed, "status": "missing_grad_l2"})
            continue
        if not isinstance(grad_metrics, dict) or not grad_metrics:
            seed_metadata.append({"seed": seed, "status": "empty_grad_l2"})
            continue

        q_metrics = sorted([name for name in grad_metrics if qname_pattern.lower() in name.lower()])
        c_metrics = sorted([name for name in grad_metrics if qname_pattern.lower() not in name.lower()])
        all_metrics = sorted(grad_metrics.keys())

        seed_quantum = _aggregate_by_step_rms(grad_metrics, q_metrics)
        seed_classical = _aggregate_by_step_rms(grad_metrics, c_metrics)
        seed_global = _aggregate_by_step_rms(grad_metrics, all_metrics)
        if seed_global is None and seed_quantum is None and seed_classical is None:
            seed_metadata.append({"seed": seed, "status": "no_usable_curves"})
            continue

        valid_lengths = [arr.size for arr in [seed_global, seed_quantum, seed_classical] if arr is not None]
        min_len = min([steps.size] + valid_lengths)
        if min_len == 0:
            seed_metadata.append({"seed": seed, "status": "zero_length"})
            continue

        step_curves.append(steps[:min_len])
        if seed_global is not None:
            global_curves.append(seed_global[:min_len])
        if seed_quantum is not None:
            quantum_curves.append(seed_quantum[:min_len])
        if seed_classical is not None:
            classical_curves.append(seed_classical[:min_len])

        seed_metadata.append(
            {
                "seed": seed,
                "status": "ok",
                "param_groups": {
                    "quantum": q_metrics,
                    "classical": c_metrics,
                },
            }
        )

    if not step_curves:
        return None, None, None, {"seeds": seed_metadata, "aggregation": "rms"}

    shared_len = min(curve.size for curve in step_curves)
    shared_steps = step_curves[0][:shared_len].astype(int)

    def _mean_curve(curves: list[np.ndarray]) -> ScalarSeries | None:
        if not curves:
            return None
        min_len = min(shared_len, min(curve.size for curve in curves))
        if min_len == 0:
            return None
        aligned = np.vstack([curve[:min_len] for curve in curves])
        return ScalarSeries(steps=shared_steps[:min_len], values=aligned.mean(axis=0))

    metadata = {
        "seeds": seed_metadata,
        "aggregation": "rms",
    }
    return _mean_curve(global_curves), _mean_curve(quantum_curves), _mean_curve(classical_curves), metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        action="append",
        required=True,
        help="Reserved for legacy CLI compatibility. Use programmatic API with loaded model dictionaries.",
    )
    parser.add_argument("--qname_pattern", default="qlayer", help="Substring used to identify quantum parameter names.")
    parser.add_argument("--out_dir", default="outputs/figures", help="Output directory for figure + JSON sidecar.")
    parser.add_argument("--format", choices=["png", "pdf"], default="pdf", help="Figure format.")
    return parser



def generate_arch_gradient_norms(
    models: Mapping[str, Mapping[int, object]],
    out_dir: str | Path,
    qname_pattern: str = "qlayer",
    output_format: str = "pdf",
) -> tuple[Path, Path]:
    if not models:
        raise ValueError("No models were provided.")

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sidecar: dict[str, object] = {
        "qname_pattern": qname_pattern,
        "models": [],
        "aggregation": "rms",
    }

    plotted_any = False
    for model_name, seed_dict in models.items():
        global_series, quantum_series, classical_series, meta = _resolve_gradient_series(
            seed_dict=seed_dict,
            qname_pattern=qname_pattern,
        )

        model_summary = {
            "model_name": model_name,
            "resolved": meta,
            "num_points": {
                "global": int(global_series.values.size) if global_series is not None else 0,
                "quantum": int(quantum_series.values.size) if quantum_series is not None else 0,
                "classical": int(classical_series.values.size) if classical_series is not None else 0,
            },
        }
        sidecar["models"].append(model_summary)

        label = model_name
        print(
            f"[{label}] "
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
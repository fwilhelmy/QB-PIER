from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _rms_stack(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a list of 1D arrays (aligned by truncation), returns (mean, std).
    """
    if not curves:
        raise ValueError("No curves to aggregate.")
    min_len = min(c.size for c in curves)
    if min_len <= 0:
        raise ValueError("Curves have zero length.")
    aligned = np.vstack([c[:min_len] for c in curves]).astype(float)
    return aligned.mean(axis=0), aligned.std(axis=0)


def _seed_rms_by_group(grad_dict: dict, keys: Sequence[str]) -> np.ndarray | None:
    """
    RMS across parameter-group curves for a single seed.
    Returns 1D array aligned by truncation, or None if nothing available.
    """
    available = [k for k in keys if k in grad_dict]
    if not available:
        return None
    min_len = min(np.asarray(grad_dict[k]).size for k in available)
    if min_len <= 0:
        return None
    stacked = np.vstack([np.asarray(grad_dict[k], dtype=float)[:min_len] for k in available])
    return np.sqrt(np.mean(stacked**2, axis=0))


def gradient_norms(
    models: Mapping[str, Mapping[int, object]],
    out_dir: str | Path = "logs",
    *,
    qname_pattern: str = "qlayer",
    include: tuple[str, ...] = ("global", "quantum", "classical"),
    filename: str = "gradient_norms.png",
) -> Path:
    """
    Plot gradient-norm evolution (global / quantum / classical) vs STEP.

    Data expected per seed model:
      - model.metrics["grad_l2"] is a dict[str, list/np.ndarray]
      - model.metrics["grad_l2/step"] is list/np.ndarray of steps

    Aggregation:
      - Per seed: RMS across curves in each group (global/quantum/classical)
      - Across seeds: mean ± std (simple, readable, good enough for overview plots)

    Saves one PNG and returns its path.
    """
    if not models:
        raise ValueError("No models provided.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    plotted_any = False

    for model_name, seed_dict in models.items():
        if not seed_dict:
            continue

        # Accumulate per-seed group curves
        steps_curves: list[np.ndarray] = []
        curves_global: list[np.ndarray] = []
        curves_quantum: list[np.ndarray] = []
        curves_classical: list[np.ndarray] = []

        for seed, model in seed_dict.items():
            grad = model.metrics["grad_l2"]
            steps = model.metrics["grad_l2/step"]

            if grad is None or steps is None:
                continue
            if not isinstance(grad, dict) or not grad:
                continue

            steps = np.asarray(steps, dtype=float)

            all_keys = sorted(grad.keys())
            q_keys = [k for k in all_keys if qname_pattern.lower() in k.lower()]
            c_keys = [k for k in all_keys if qname_pattern.lower() not in k.lower()]

            seed_global = _seed_rms_by_group(grad, all_keys) if "global" in include else None
            seed_quantum = _seed_rms_by_group(grad, q_keys) if "quantum" in include else None
            seed_classical = _seed_rms_by_group(grad, c_keys) if "classical" in include else None

            candidates = [arr for arr in (seed_global, seed_quantum, seed_classical) if arr is not None]
            if not candidates:
                continue

            min_len = min([steps.size] + [a.size for a in candidates])
            if min_len <= 0:
                continue

            steps_curves.append(steps[:min_len])
            if seed_global is not None:
                curves_global.append(seed_global[:min_len])
            if seed_quantum is not None:
                curves_quantum.append(seed_quantum[:min_len])
            if seed_classical is not None:
                curves_classical.append(seed_classical[:min_len])

        if not steps_curves:
            continue

        # Align all seeds to a shared length (common prefix) so mean/std makes sense
        shared_len = min(s.size for s in steps_curves)
        x = steps_curves[0][:shared_len]

        def plot_group(curves: list[np.ndarray], label: str, linestyle: str = "-", lw: float = 2.0):
            nonlocal plotted_any
            if not curves:
                return
            mean, std = _rms_stack([c[:shared_len] for c in curves])
            ax.plot(x, mean, linestyle=linestyle, linewidth=lw, label=label)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15)
            plotted_any = True

        plot_group(curves_global,   f"{model_name} — global",   linestyle="-",  lw=2.0)
        plot_group(curves_quantum,  f"{model_name} — quantum",  linestyle="--", lw=1.8)
        plot_group(curves_classical,f"{model_name} — classical",linestyle=":",  lw=1.8)

    if not plotted_any:
        raise RuntimeError("Nothing was plotted (missing grad_l2 / grad_l2/step across all models).")

    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient norm (RMS)")
    ax.set_title("Gradient norm evolution")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path

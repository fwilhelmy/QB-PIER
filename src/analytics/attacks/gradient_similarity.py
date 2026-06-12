"""White-box gradient-similarity analytics between models.

Pairwise similarity of FGSM input gradients, exposed through the same
configuration system as the attacksv2 transfer views (grain = family /
architecture / variant, mean/best aggregation, per-view blacklist,
row/column selection, asymmetric grains).

Workflow
--------
1. ``scripts/compute_gradient_similarity_cache.py`` (cluster) loads every
   trained model, computes input gradients over ``N_SAMPLES`` images, and
   caches per-model tensors + the long-form pair-metrics CSV under
   ``logs/GRADIENTS/``.
2. ``notebooks/attacks/gradient_similarity.ipynb`` loads the cached CSV and
   builds matrices/figures/rankings with :data:`GRADSIM_TEMPLATES` or any
   custom :class:`AttacksV2Config`.

Pair metrics are computed for ALL models (blacklisted ones included); the
blacklist is applied per-view through the config, exactly like attacksv2.

Metric semantics: every gradient-similarity metric is "higher = more
similar".  They are registered with ``METRIC_HIGHER_IS_DAMAGE = True`` so the
shared aggregation backend's ``best_damage`` mode means "most similar" and
``best_robust`` means "least similar"; the aliases ``most_similar`` and
``least_similar`` are accepted wherever an aggregate mode is expected.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

try:  # plotting is optional for table-only use
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]

from analytics.paths import find_project_root, output_dir
from analytics.attacks.attacksv2_transfer_tools import AttacksV2Config, export_table
from analytics.attacks.transfer_role_tools import (
    METRIC_ALIASES,
    METRIC_HIGHER_IS_DAMAGE,
    build_attack_matrix,
    build_top_sources_by_target,
    plot_attack_matrix,
    save_figure,
)

log = logging.getLogger(__name__)

__all__ = [
    "GRADSIM_METRICS",
    "GRADSIM_TEMPLATES",
    "GRADSIM_FIGURE_TEMPLATES",
    "DEFAULT_GRADIENT_CACHE_DIR",
    "safe_model_name",
    "compute_pair_metrics",
    "compute_pair_metrics_from_cache",
    "load_gradient_pair_metrics",
    "get_gradsim_template",
    "run_gradient_similarity_report",
]

#: Display names of the similarity metrics (columns of the pair-metrics CSV).
GRADSIM_METRICS: tuple[str, ...] = (
    "Sign Agreement",
    "Gradient Cosine",
    "Saliency TopK Jaccard",
    "Saliency Pearson",
)

# Register the similarity metrics with the shared transfer backend so that
# resolve_metric / metric_higher_is_damage / aggregation modes work unchanged.
for _metric in GRADSIM_METRICS:
    METRIC_ALIASES.setdefault(_metric, _metric)
    METRIC_ALIASES.setdefault(_metric.lower(), _metric)
    METRIC_ALIASES.setdefault(_metric.lower().replace(" ", "_"), _metric)
    METRIC_HIGHER_IS_DAMAGE.setdefault(_metric, True)

#: Friendly aggregate-mode aliases for similarity metrics.
_SIMILARITY_AGG_ALIASES = {"most_similar": "best_damage", "least_similar": "best_robust"}

DEFAULT_GRADIENT_CACHE_DIR = "logs/GRADIENTS"
PAIR_METRICS_FILENAME = "gradient_pair_metrics.csv"
GRADIENTS_SUBDIR = "gradients"


def safe_model_name(name: str) -> str:
    """Filename-safe model name (same scheme as the attack pipeline)."""
    return (
        str(name)
        .replace("\\", "__")
        .replace("/", "__")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _translate_similarity_modes(overrides: dict[str, object]) -> dict[str, object]:
    for key in ("variant_aggregate_mode", "architecture_aggregate_mode"):
        value = overrides.get(key)
        if isinstance(value, str) and value in _SIMILARITY_AGG_ALIASES:
            overrides[key] = _SIMILARITY_AGG_ALIASES[value]
    return overrides


# ---------------------------------------------------------------------------
# Templates ("suggested best configurations")
# ---------------------------------------------------------------------------

GRADSIM_TEMPLATES: dict[str, AttacksV2Config] = {
    # Full variant x variant matrix. CSV only: the heatmap is unreadable at
    # this grain (too many variants).
    "variant_full": AttacksV2Config(
        source_grain="variant",
        target_grain="variant",
        comparison_mode="inter_variant",
        metric="Sign Agreement",
        apply_global_blacklist=False,
        variant_aggregate_mode="mean",
    ),
    # Architecture x architecture, mean over variant pairs. Diagonal cells show
    # intra-architecture (cross-variant) similarity.
    "architecture_mean": AttacksV2Config(
        source_grain="architecture",
        target_grain="architecture",
        comparison_mode="inter_variant",
        metric="Sign Agreement",
        apply_global_blacklist=False,
        variant_aggregate_mode="mean",
    ),
    # Architecture x architecture where each cell shows the MOST similar
    # variant pair (best_damage == most similar for similarity metrics).
    "architecture_most_similar": AttacksV2Config(
        source_grain="architecture",
        target_grain="architecture",
        comparison_mode="inter_variant",
        metric="Sign Agreement",
        apply_global_blacklist=False,
        variant_aggregate_mode="best_damage",
    ),
    # Family x family synthesis.
    "family_mean": AttacksV2Config(
        source_grain="family",
        target_grain="family",
        comparison_mode="inter_variant",
        metric="Sign Agreement",
        apply_global_blacklist=False,
        variant_aggregate_mode="mean",
        architecture_aggregate_mode="mean",
    ),
    # Architecture grain with quantum/hybrid ansatz split out (VQM-Strong, ...).
    "ansatz_split": AttacksV2Config(
        source_grain="architecture",
        target_grain="architecture",
        comparison_mode="inter_variant",
        metric="Sign Agreement",
        apply_global_blacklist=False,
        variant_aggregate_mode="mean",
        split_quantum_ansatz_as_architecture=True,
    ),
    # Asymmetric view: quantum variants as rows, every architecture as columns.
    "quantum_variants_vs_architectures": AttacksV2Config(
        source_grain="variant",
        target_grain="architecture",
        comparison_mode="inter_variant",
        metric="Sign Agreement",
        apply_global_blacklist=False,
        variant_aggregate_mode="mean",
        source_family_filter="Quantum",
    ),
}

#: Templates whose matrices are also rendered as heatmap figures.
GRADSIM_FIGURE_TEMPLATES: frozenset[str] = frozenset(
    {
        "architecture_mean",
        "architecture_most_similar",
        "family_mean",
        "ansatz_split",
        "quantum_variants_vs_architectures",
    }
)


def get_gradsim_template(name: str, **overrides: object) -> AttacksV2Config:
    if name not in GRADSIM_TEMPLATES:
        allowed = ", ".join(sorted(GRADSIM_TEMPLATES))
        raise KeyError(f"Unknown gradient-similarity template {name!r}. Available: {allowed}")
    return GRADSIM_TEMPLATES[name].with_updates(**_translate_similarity_modes(dict(overrides)))


# ---------------------------------------------------------------------------
# Pair-metric computation (torch imported lazily: views only need pandas)
# ---------------------------------------------------------------------------

def compute_pair_metrics(
    grads_by_model: Mapping[str, "object"],
    *,
    top_k_fraction: float = 0.1,
) -> pd.DataFrame:
    """Long-form pairwise gradient-similarity metrics between models.

    ``grads_by_model`` maps model name -> tensor of flattened per-sample input
    gradients ``[N, D]`` (any float dtype). Every model is compared with every
    other over the shared sample axis; rows are emitted in both directions
    (and for the self pair) so the result behaves like the transfer pair
    dataframes.

    Metrics (mean over samples; all "higher = more similar"):

    - ``Sign Agreement``: fraction of pixels with equal ``sign(grad)``.
      Most FGSM-relevant (the attack only uses the sign); random ~0.5.
    - ``Gradient Cosine``: cosine of the flattened raw gradients.
    - ``Saliency TopK Jaccard``: overlap of the top ``top_k_fraction`` pixels
      by ``|grad|`` (do the models attack the same image regions?).
    - ``Saliency Pearson``: Pearson correlation of the ``|grad|`` maps.
    """
    import torch

    names = sorted(str(n) for n in grads_by_model)
    grads = torch.stack(
        [torch.as_tensor(grads_by_model[name]).to(torch.float32) for name in names]
    )  # [M, N, D]
    n_models, n_samples, dim = grads.shape
    tiny = 1e-12

    signs = grads.sign()
    unit = grads / grads.norm(dim=-1, keepdim=True).clamp_min(tiny)
    sal = grads.abs()
    top_k = max(1, int(round(top_k_fraction * dim)))
    topk_idx = sal.topk(top_k, dim=-1).indices
    masks = torch.zeros_like(sal, dtype=torch.bool).scatter_(-1, topk_idx, True)
    sal_centered = sal - sal.mean(dim=-1, keepdim=True)
    sal_unit = sal_centered / sal_centered.norm(dim=-1, keepdim=True).clamp_min(tiny)

    records: list[dict[str, object]] = []
    for a in range(n_models):
        for b in range(a, n_models):
            if a == b:
                values = {metric: 1.0 for metric in GRADSIM_METRICS}
            else:
                values = {
                    "Sign Agreement": (signs[a] == signs[b]).float().mean().item(),
                    "Gradient Cosine": (unit[a] * unit[b]).sum(dim=-1).mean().item(),
                    "Saliency TopK Jaccard": (
                        (masks[a] & masks[b]).sum(dim=-1).float()
                        / (masks[a] | masks[b]).sum(dim=-1).float().clamp_min(1)
                    ).mean().item(),
                    "Saliency Pearson": (sal_unit[a] * sal_unit[b]).sum(dim=-1).mean().item(),
                }
            pairs = [(a, b)] if a == b else [(a, b), (b, a)]
            for i, j in pairs:
                record: dict[str, object] = {
                    "Source Model": names[i],
                    "Target Model": names[j],
                    "n_samples": int(n_samples),
                    "top_k_fraction": float(top_k_fraction),
                }
                record.update(values)
                records.append(record)

    return pd.DataFrame.from_records(records)


def compute_pair_metrics_from_cache(
    cache_dir: str | Path = DEFAULT_GRADIENT_CACHE_DIR,
    *,
    top_k_fraction: float = 0.1,
    write_csv: bool = True,
) -> pd.DataFrame:
    """Rebuild the pair-metrics CSV from cached per-model gradient tensors.

    Expects ``<cache_dir>/gradients/<safe_name>.pt`` files written by
    ``scripts/compute_gradient_similarity_cache.py``; no model loading needed.
    """
    import torch

    cache_dir = _resolve_cache_dir(cache_dir)
    grad_dir = cache_dir / GRADIENTS_SUBDIR
    files = sorted(grad_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(
            f"No cached gradients under {grad_dir}. "
            "Run scripts/compute_gradient_similarity_cache.py first."
        )

    grads_by_model: dict[str, object] = {}
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        grads_by_model[str(payload["model_name"])] = payload["gradients"]

    pair_df = compute_pair_metrics(grads_by_model, top_k_fraction=top_k_fraction)
    if write_csv:
        out_path = cache_dir / PAIR_METRICS_FILENAME
        pair_df.to_csv(out_path, index=False)
        log.info("Wrote %d pair rows for %d models to %s", len(pair_df), len(files), out_path)
    return pair_df


def _resolve_cache_dir(cache_dir: str | Path) -> Path:
    path = Path(cache_dir)
    if not path.is_absolute():
        path = find_project_root() / path
    return path


def load_gradient_pair_metrics(
    cache_dir: str | Path = DEFAULT_GRADIENT_CACHE_DIR,
    *,
    rebuild: bool = False,
    top_k_fraction: float = 0.1,
) -> pd.DataFrame:
    """Load the cached gradient pair-metrics CSV (``rebuild`` recomputes from .pt files)."""
    cache_dir = _resolve_cache_dir(cache_dir)
    csv_path = cache_dir / PAIR_METRICS_FILENAME
    if rebuild or not csv_path.exists():
        if not rebuild and not (cache_dir / GRADIENTS_SUBDIR).exists():
            raise FileNotFoundError(
                f"Gradient pair metrics not found at {csv_path} and no gradient cache to "
                "rebuild from. Run scripts/compute_gradient_similarity_cache.py "
                "(on the cluster) and sync the cache directory back."
            )
        return compute_pair_metrics_from_cache(cache_dir, top_k_fraction=top_k_fraction)
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Standard report
# ---------------------------------------------------------------------------

def run_gradient_similarity_report(
    pair_df: pd.DataFrame,
    *,
    out_root: str | Path | None = None,
    metrics: Sequence[str] = GRADSIM_METRICS,
    templates: Mapping[str, AttacksV2Config] | None = None,
    figure_templates: frozenset[str] | None = None,
    ranking_top_k: int = 3,
) -> dict[str, Path]:
    """Export matrices, heatmaps, and similarity rankings for every metric.

    Per metric and template: matrix CSV (all templates) and heatmap PNG/PDF
    (figure templates only -- never variant x variant, which is unreadable).
    Additionally, per metric at architecture grain: top-k most / least similar
    source groups for each target group.
    """
    if out_root is None:
        out_root = output_dir(find_project_root(), "gradient_similarity")
    out_root = Path(out_root)
    templates = dict(templates if templates is not None else GRADSIM_TEMPLATES)
    figure_set = figure_templates if figure_templates is not None else GRADSIM_FIGURE_TEMPLATES

    outputs: dict[str, Path] = {}
    for metric in metrics:
        metric_slug = metric.lower().replace(" ", "_")
        metric_dir = out_root / metric_slug

        for name, template in templates.items():
            cfg = template.with_updates(metric=metric)
            matrix = build_attack_matrix(pair_df, cfg)
            paths = export_table(matrix, metric_dir, f"{name}__{cfg.slug()}", index=True)
            outputs[f"{metric_slug}/{name}/csv"] = paths["csv"]
            if name in figure_set and plt is not None:
                fig, _ = plot_attack_matrix(matrix, title=f"{metric} - {name}")
                fig_path = save_figure(fig, metric_dir / f"{name}__{cfg.slug()}.png")
                plt.close(fig)
                outputs[f"{metric_slug}/{name}/figure"] = fig_path

        rank_cfg = get_gradsim_template("architecture_mean", metric=metric, top_k=ranking_top_k)
        for rank_mode, label in (("damage", "most_similar"), ("robust", "least_similar")):
            wide, long = build_top_sources_by_target(pair_df, rank_cfg, rank_mode=rank_mode)
            paths = export_table(wide, metric_dir, f"{label}_sources_per_target")
            outputs[f"{metric_slug}/{label}/csv"] = paths["csv"]
            export_table(long, metric_dir, f"{label}_sources_per_target_long")
        log.info("Gradient similarity report written for %s under %s", metric, metric_dir)

    return outputs

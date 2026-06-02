import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def cell_effectiveness_max_steepness(subdf: pd.DataFrame) -> float:
    """
    Attack effectiveness for a single (source, target) pair:
    maximum accuracy drop per epsilon step.

    subdf: rows for fixed (source_model, target_model) with:
           'epsilon', 'robust_accuracy'.

    Returns:
        scalar >= 0. Larger => sharper worst-case drop (more vulnerable).
    """
    if subdf.empty:
        return 0.0

    subdf = subdf.sort_values("epsilon")
    eps = subdf["epsilon"].to_numpy(float)
    acc = subdf["robust_accuracy"].to_numpy(float)

    if eps.size < 2:
        return 0.0

    max_slope = 0.0
    for i in range(1, len(eps)):
        de = eps[i] - eps[i - 1]
        if de <= 0:
            continue

        da = acc[i - 1] - acc[i]  # positive if accuracy drops
        if da > 0:
            max_slope = max(max_slope, da / de)

    return float(max_slope)


def cell_effectiveness_max_rel_drop(subdf: pd.DataFrame) -> float:
    """
    Max relative accuracy drop at max epsilon
    “Given our strongest attack in this sweep, how much do you suffer vs clean?”

    (clean_acc - min_acc) / clean_acc

    Higher => huge final damage given max epsilon.
    """
    if subdf.empty:
        return 0.0

    subdf = subdf.sort_values("epsilon")
    acc = subdf["robust_accuracy"].to_numpy(float)

    if acc.size == 0:
        return 0.0

    clean_acc = acc[0]
    if clean_acc <= 0:
        return 0.0

    min_acc = float(acc.min())
    rel_drop = (clean_acc - min_acc) / clean_acc
    return float(max(rel_drop, 0.0))

def cell_effectiveness_auc_drop(subdf: pd.DataFrame) -> float:
    """
    Total robustness loss (AUC-based)
    “How much accuracy do we lose across the whole epsilon range?”
    1 - (AUC(actual accuracy curve) / AUC(ideal flat-at-clean curve)).

    Higher => less robust over the whole epsilon range.
    
    """
    if subdf.empty:
        return 0.0

    subdf = subdf.sort_values("epsilon")
    eps = subdf["epsilon"].to_numpy(float)
    acc = subdf["accuracy"].to_numpy(float)

    if eps.size < 2:
        return 0.0

    clean_acc = acc[0]
    ideal_auc = clean_acc * (eps[-1] - eps[0])
    if ideal_auc <= 0:
        return 0.0

    actual_auc = np.trapz(acc, eps)
    drop = 1.0 - (actual_auc / ideal_auc)
    return float(max(drop, 0.0))


def cell_effectiveness_eps_to_fraction(subdf: pd.DataFrame, frac: float = 0.5) -> float:
    """
    Threshold fragility (epsilon at which it collapses)
    “How quickly can I push you below a given fraction of clean accuracy?”
    
    Invert epsilon where accuracy first <= frac * clean_acc.
    If it never drops below: 0 (robust).
    Higher => breaks earlier => more vulnerable.
    """
    if subdf.empty:
        return 0.0

    subdf = subdf.sort_values("epsilon")
    eps = subdf["epsilon"].to_numpy(float)
    acc = subdf["accuracy"].to_numpy(float)

    if eps.size < 2:
        return 0.0

    clean_acc = acc[0]
    if clean_acc <= 0:
        return 0.0

    thresh = frac * clean_acc
    for e, a in zip(eps[1:], acc[1:]):
        if a <= thresh:
            # bigger score if it fails at small epsilon
            return float(1.0 / max(e, 1e-8))

    return 0.0


def plot_model_attack_effectiveness_matrix(
    df: pd.DataFrame,
    out_dir: str,
    cell_fn=cell_effectiveness_max_steepness,
    normalize: bool = True,
    cmap: str = "viridis",
    fname: str = "attack_effectiveness_matrix.png",
    annotate: bool = True,
) -> None:
    """
    Build and plot a matrix of attack effectiveness between models.

    Rows : adversarial source models
    Cols : evaluated target models
    Cell : cell_fn(df_pair) for that (source, target)

    If normalize=True, divide all positive values by the global max so the
    strongest vulnerability = 1.0.
    """
    os.makedirs(out_dir, exist_ok=True)

    sources = sorted(df["source_model"].unique())
    targets = sorted(df["target_model"].unique())

    n_src, n_tgt = len(sources), len(targets)
    mat = np.zeros((n_src, n_tgt), dtype=float)

    # fill matrix
    for i, s in enumerate(sources):
        for j, t in enumerate(targets):
            subdf = df[(df["source_model"] == s) & (df["target_model"] == t)]
            mat[i, j] = cell_fn(subdf)

    # normalise
    if normalize:
        max_val = np.max(mat)
        mat_norm = mat / max_val if max_val > 0 else mat
    else:
        mat_norm = mat

    fig, ax = plt.subplots(figsize=(1.2 + 0.5 * n_tgt, 1.2 + 0.5 * n_src))

    im = ax.imshow(
        mat_norm,
        interpolation="nearest",
        cmap=cmap,
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
    )

    # use model names directly, but no extra "source/target" axis labels
    ax.set_xticks(range(n_tgt))
    ax.set_xticklabels([str(t) for t in targets], rotation=45, ha="right")

    ax.set_yticks(range(n_src))
    ax.set_yticklabels([str(s) for s in sources])

    ax.set_title("Model×Model attack effectiveness (normalised)", fontsize=10)

    # light grid to make the matrix easier to read
    ax.set_xticks(np.arange(-0.5, n_tgt, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_src, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.3, alpha=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    # annotate cells with numeric values (suppress tiny ones)
    if annotate:
        for i in range(n_src):
            for j in range(n_tgt):
                val = mat_norm[i, j]
                if val > 0.03:  # threshold to avoid visual clutter
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalised vulnerability", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=220)
    plt.close(fig)

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pandas as pd


def plot_tp_heatmaps_by_class(
    df: pd.DataFrame,
    y_true: torch.Tensor,
    classes,
    root_dir: str,
    models_name: list[str],
    cmap: str = "viridis",
) -> None:
    """
    For each (source_model, target_model) pair, create a heatmap:

        rows:    true classes
        cols:    epsilons
        values:  TP% for that class at that epsilon
    """
    os.makedirs(root_dir, exist_ok=True)

    # Ensure numpy array for indexing
    y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else np.asarray(y_true)
    num_classes = len(classes)

    # Iterate over model pairs
    for model_name, ((source_model, target_model), subdf_pair) in zip(models_name, df.loc[df['source_model'] == df['target_model']].groupby(
        ["source_model", "target_model"]
    )):
        # Sort epsilons consistently
        epsilons = sorted(subdf_pair["epsilon"].unique())
        num_eps = len(epsilons)

        # Heatmap: [num_classes x num_eps]
        tp_matrix = np.zeros((num_classes, num_eps), dtype=float)

        # Precompute per-class counts
        class_counts = np.array([(y_true == c).sum() for c in range(num_classes)])

        for j, eps in enumerate(epsilons):
            sub = subdf_pair[subdf_pair["epsilon"] == eps]

            if sub.empty:
                tp_matrix[:, j] = np.nan
                continue

            # Expect a single row; if multiple, take the first
            preds = sub.iloc[0]["preds"]
            preds = preds.cpu().numpy() if torch.is_tensor(preds) else np.asarray(preds)

            for c in range(num_classes):
                mask = (y_true == c)
                n_c = class_counts[c]
                if n_c == 0:
                    tp = np.nan
                else:
                    tp = (preds[mask] == c).sum() / float(n_c) * 100.0
                tp_matrix[c, j] = tp

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(1.5 + 0.7 * num_eps, 1.5 + 0.5 * num_classes))
        im = ax.imshow(tp_matrix, aspect="auto", origin="upper", cmap=cmap)

        ax.set_xticks(range(num_eps))
        ax.set_xticklabels([f"{e:g}" for e in epsilons], rotation=45, ha="right")
        ax.set_xlabel("FGSM epsilon")

        ax.set_yticks(range(num_classes))
        ax.set_yticklabels(classes)
        ax.set_ylabel("True class")

        title = f"TP% by class vs epsilon\nmodel={model_name}"
        ax.set_title(title, fontsize=10)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("True Positive rate (%)")

        fig.tight_layout()

        outfile = os.path.join(
            root_dir,
            f"tp_heatmap_{model_name}.png",
        )
        fig.savefig(outfile, dpi=200)
        plt.close(fig)

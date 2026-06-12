"""Saliency and perturbation analyticss."""

from __future__ import annotations

import logging
import os
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.utilities import rank_zero_only

from attacks import AttackFactory
from attacks.FGSM import FGSM
from datasets import DatasetFactory
from loading import load_experiment

from analytics.attacks._eval_helpers import LimitedDataset, ensure_model_metadata
from analytics.attacks.attack_matrix import parse_canonical_model_name
from analytics.attacks.model_blacklist import model_name_is_blacklisted

__all__ = [
    "plot_perturbation_and_saliency",
    "run_perturbation_saliency_for_models",
]

log = logging.getLogger(__name__)

_INVALID_FILENAME_CHARS = '<>:"/\\|?*'


def _sanitize_filename(name: str) -> str:
    return "".join("-" if c in _INVALID_FILENAME_CHARS else c for c in name)


@rank_zero_only
def plot_perturbation_and_saliency(
    model,
    attack: FGSM,
    device: torch.device,
    epsilon: float,
    out_dir,
    k: int = 2,
    save_path_fn: Callable[[int], str] | None = None,
):
    model.eval()
    x = attack.dataset.data.to(device)
    grads = attack.gradients[model.log_dir]
    n = min(k, len(grads), x.size(0))
    for i in range(n):
        image_t = x[i].detach().cpu()
        image = image_t.squeeze().numpy()
        saliency = grads[i][1].detach().cpu()
        saliency_sign = saliency.sign().numpy()
        perturbed = attack._fgsm(image_t, saliency, epsilon).squeeze().numpy()

        # Normalise saliency for visualising overlay
        saliency_abs = saliency.abs()
        if saliency_abs.max() > 0:
            saliency_overlay = (saliency_abs / saliency_abs.max()).numpy()
        else:
            saliency_overlay = saliency_abs.numpy()

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(saliency_sign.squeeze(), cmap="hot")
        axes[1].set_title("Saliency")
        axes[1].axis("off")
        axes[2].imshow(image, cmap="gray")
        axes[2].imshow(saliency_overlay.squeeze(), cmap="viridis", alpha=0.6)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
        axes[3].imshow(perturbed, cmap="gray")
        axes[3].set_title(f"Perturbed (eps={epsilon:g})")
        axes[3].axis("off")
        plt.tight_layout()
        path = save_path_fn(i) if save_path_fn else os.path.join(out_dir, f"saliency_{i}.png")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        plt.savefig(path)
        plt.close(fig)


def run_perturbation_saliency_for_models(
    model_logdir: str,
    output_dir: str,
    *,
    k: int = 2,
    models: Sequence[str] | None = None,
    epsilons: Sequence[float] = (0.3,),
    exclude_blacklisted: bool = False,
    max_dataset_samples: int = 128,
    batch_size: int = 32,
    dataset_name: str = "mnist",
    dataset_root: str = "./data/processed",
) -> None:
    """Generate ``k`` saliency figures per epsilon for each selected model.

    ``models=None`` selects every trained model under ``model_logdir``
    (including blacklisted ones; set ``exclude_blacklisted=True`` to apply the
    shared attack model blacklist). Figures are laid out for cross-model
    comparison: ``output_dir/eps=<eps>/example_<i>/<family>/<arch>_<variant>.png``.
    """
    os.makedirs(output_dir, exist_ok=True)
    trained_models = load_experiment(model_logdir, load="final")

    if models is None:
        selected = [
            name
            for name in trained_models
            if not (exclude_blacklisted and model_name_is_blacklisted(name))
        ]
    else:
        missing = sorted(set(models) - set(trained_models))
        if missing:
            log.warning("Requested models not found in %s: %s", model_logdir, missing)
        selected = [name for name in models if name in trained_models]

    if not selected:
        log.warning("No models selected for perturbation/saliency figures")
        return

    try:
        dm = DatasetFactory.create(dataset_name, root=dataset_root, train=False)
        full_dataset = dm.dataset
        classes = getattr(full_dataset, "classes", [str(i) for i in range(10)])

        limit = min(max_dataset_samples, len(full_dataset))
        data_samples, labels = [], []
        for idx in range(limit):
            sample, label = full_dataset[idx]
            sample_t = sample if isinstance(sample, torch.Tensor) else torch.as_tensor(sample)
            data_samples.append(sample_t)
            labels.append(int(label))

        dataset = LimitedDataset(torch.stack(tuple(data_samples)), torch.tensor(labels), classes)
    except Exception as exc:
        log.warning("Failed to load dataset %s: %s", dataset_name, exc)
        return

    fgsm = AttackFactory.create("fgsm", dataset=dataset, batch_size=min(batch_size, len(dataset)))

    for model_name in selected:
        model_seeds = trained_models.get(model_name) or {}
        if not model_seeds:
            log.warning("No seeds found for model %s; skipping", model_name)
            continue
        model = next(iter(model_seeds.values()))
        ensure_model_metadata(model_name, model, model_logdir)
        model.eval()
        fgsm.compute_gradients(model)

        family, arch, variant = parse_canonical_model_name(model_name)
        fname = _sanitize_filename(f"{arch}_{variant}.png" if variant else f"{arch}.png")
        family_dir = _sanitize_filename(family)
        for epsilon in epsilons:
            eps_dir = os.path.join(str(output_dir), f"eps={epsilon:g}")

            def save_path_fn(i: int, _eps_dir=eps_dir, _family=family_dir, _fname=fname) -> str:
                return os.path.join(_eps_dir, f"example_{i}", _family, _fname)

            plot_perturbation_and_saliency(
                model, fgsm, torch.device("cpu"), epsilon, eps_dir, k=k, save_path_fn=save_path_fn
            )
            log.info("Saved saliency figures for %s (eps=%g) under %s", model_name, epsilon, eps_dir)

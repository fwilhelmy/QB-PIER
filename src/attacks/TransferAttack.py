"""Utility for running transfer attacks between two models."""

from __future__ import annotations

from typing import Iterable, List, Dict

import pandas as pd
import torch

from .FGSMAttack import FGSMAttack
from src.data.DatasetManagerFactory import DatasetManagerFactory
from utils import calculate_accuracy


def run_transfer_attack(
    attacker_model,
    victim_model,
    dataset_name: str,
    device: torch.device | str = "cpu",
    epsilons: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Execute a transfer attack from ``attacker_model`` to ``victim_model``.

    Parameters
    ----------
    attacker_model, victim_model : torch.nn.Module
        The models used to craft and evaluate adversarial examples.
    dataset_name : str
        Name of the dataset used for computing the attack.
    device : torch.device or str, optional
        Device on which to perform the computations.
    epsilons : Iterable[float], optional
        List of epsilon values to evaluate. Defaults to ``[0.0, 0.05, ..., 1.0]``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``epsilon`` and ``accuracy`` containing the
        victim's accuracy for each epsilon level.
    """

    # Prepare dataset and gradients once for efficiency
    dm = DatasetManagerFactory(dataset_name)
    dataset = dm.load_dataset(train=False)
    dm.dataset = dataset
    data = dataset.data.unsqueeze(1).to(torch.float)
    targets = dataset.targets

    gradients = dm.compute_gradients(attacker_model)
    attack = FGSMAttack()
    epsilons = list(epsilons) if epsilons is not None else [x / 100.0 for x in range(0, 105, 5)]

    attacker_model.eval()
    victim_model.eval()

    records: List[Dict[str, float]] = []

    with torch.no_grad():
        attacker_model(data.to(device))  # warm-up forward pass

    for eps in epsilons:
        adv_examples = attack.attack(gradients, eps).to(device)
        with torch.no_grad():
            adv_logits = victim_model(adv_examples)
            adv_acc = calculate_accuracy(
                adv_logits,
                targets[: adv_examples.size(0)],
            )
        records.append({"epsilon": eps, "accuracy": float(adv_acc)})

    return pd.DataFrame(records)

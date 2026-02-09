"""Dataset information helpers."""

from __future__ import annotations

from lightning.pytorch.utilities import rank_zero_only

__all__ = ["print_dataset_info"]

@rank_zero_only
def print_dataset_info(dataset):
    print("Dataset Information:")
    print("Number of samples: %s", len(dataset))
    if hasattr(dataset, "classes"):
        print("Number of classes: %s", len(dataset.classes))
        print("Classes: %s", dataset.classes)

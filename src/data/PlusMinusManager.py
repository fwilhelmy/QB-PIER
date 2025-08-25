'''
Dataset Source : https://pennylane.ai/datasets/plus-minus
Some code inspired from: https://pennylane.ai/qml/demos/tutorial_adversarial_attacks_QML
'''

from torch.utils.data import TensorDataset
from data.DatasetManager import DatasetManager

class PlusMinusManager(DatasetManager):
    """Dataset manager for the PennyLane PlusMinus dataset."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_dataset(self, train: bool):
        """Load the PlusMinus dataset.

        The dataset is provided by PennyLane via ``pennylane.datasets``.
        Each sample corresponds to a small image labelled as either a
        ``+`` or ``-`` symbol.
        """
        try:
            from pennylane import datasets
            try:
                data = datasets.load_plus_minus()
            except AttributeError:
                # Fallback for older PennyLane versions
                data = datasets.load_plusminus()
        except Exception as exc:  # pragma: no cover - dependency not installed
            raise ImportError(
                "PennyLane and the plus-minus dataset are required to load this dataset"
            ) from exc

        images = data["images"] if "images" in data else data["x"]
        labels = data["labels"] if "labels" in data else data["y"]

        if train and "train" in images:
            x = images["train"]
            y = labels["train"]
        elif not train and "test" in images:
            x = images["test"]
            y = labels["test"]
        else:
            x = images
            y = labels

        return TensorDataset(x, y)

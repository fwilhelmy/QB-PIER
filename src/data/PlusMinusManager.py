# See: PennyLane demo 'Adversarial attacks and robustness for QML'.
# https://pennylane.ai/qml/demos/tutorial_adversarial_attacks_QML

import pennylane as qml
import torch
from torch.utils.data import TensorDataset
from DatasetManager import DatasetManager

class PlusMinusManager(DatasetManager):
    """Dataset manager for the PennyLane PlusMinus dataset.

    Uses the PennyLane Datasets API as in the 'Adversarial attacks and robustness
    for quantum machine learning' demo.

    Dataset details:
      - 4 classes: {+, −, ⟂ (right half-plus), ⊣ (left half-plus)}
      - Images: 16x16 grayscale
      - Train Size: 1000
      - Test Size: 200
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_dataset(self, train: bool):
        [pm] = qml.data.load("other", name="plus-minus")

        x_np = pm.img_train if train else pm.img_test
        y_np = pm.labels_train if train else pm.labels_test

        # Convert to torch Tensors (match types the demo uses when moving to torch)
        x = torch.from_numpy(x_np).to(dtype=torch.float32)
        y = torch.from_numpy(y_np).to(dtype=torch.int64)

        return TensorDataset(x, y)

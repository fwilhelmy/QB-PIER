import pytest

torch = pytest.importorskip("torch")

from src.attacks.FGSM import FGSM
from tests.datasets.DummyDataset import DummyDataset
from tests.models.DummyModel import DummyModel


def test_fgsm_epsilon_variation():
    model = DummyModel()
    dataset = DummyDataset()

    fgsm = FGSM(dataset=dataset, max_samples=len(dataset))
    adv_small = fgsm.attack(model, epsilon=0.1)
    adv_large = fgsm.attack(model, epsilon=0.3)

    assert not torch.allclose(adv_small, adv_large)

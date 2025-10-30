import pytest

torch = pytest.importorskip("torch")

from src.attacks.PGD import PGD
from tests.datasets.DummyDataset import DummyDataset, DummyDatasetManager
from tests.models.DummyModel import DummyModel


def test_pgd_iterative_updates():
    model = DummyModel()
    dataset = DummyDataset()
    dm = DummyDatasetManager(dataset)

    pgd_one = PGD(
        dataset=dm.dataset,
        attack_name="PGD",
        num_iter=1,
        alpha=0.1,
        random_start=False,
    )
    adv1 = pgd_one.attack(model, epsilon=0.2)

    pgd_three = PGD(
        dataset=dm.dataset,
        attack_name="PGD",
        num_iter=3,
        alpha=0.1,
        random_start=False,
    )
    adv3 = pgd_three.attack(model, epsilon=0.2)

    assert not torch.allclose(adv1, adv3)

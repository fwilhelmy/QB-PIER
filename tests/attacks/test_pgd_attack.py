import pytest

torch = pytest.importorskip("torch")

from src.attacks.PGD import PGD


class DummyDatasetManager:
    def __init__(self, dataset):
        self.dataset = dataset

    def compute_gradients(self, model, max_x=None, hot_reload: bool = False):
        return []


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(28 * 28, 10)
        self.model_path = "dummy"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.fc(x)

    @property
    def loss_fn(self):
        return torch.nn.functional.cross_entropy


def test_pgd_iterative_updates():
    model = DummyModel()
    images = torch.zeros(1, 1, 28, 28)
    labels = torch.tensor([1])
    dataset = torch.utils.data.TensorDataset(images, labels)
    dm = DummyDatasetManager(dataset)

    pgd_one = PGD(
        dataset=dm.dataset,
        attack_name="PGD",
        num_iter=1,
        alpha=0.1,
        random_start=False,
        max_samples=len(dm.dataset),
    )
    adv1 = pgd_one.attack(model, epsilon=0.2)

    pgd_three = PGD(
        dataset=dm.dataset,
        attack_name="PGD",
        num_iter=3,
        alpha=0.1,
        random_start=False,
        max_samples=len(dm.dataset),
    )
    adv3 = pgd_three.attack(model, epsilon=0.2)

    assert not torch.allclose(adv1, adv3)

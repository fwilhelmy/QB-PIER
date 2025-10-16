import pytest

torch = pytest.importorskip("torch")

from src.attacks.FGSM import FGSM


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

    @property
    def model_name(self):
        return "dummy"


def test_fgsm_epsilon_variation():
    model = DummyModel()
    images = torch.zeros(1, 1, 28, 28)
    labels = torch.tensor([1])
    dataset = torch.utils.data.TensorDataset(images, labels)

    fgsm = FGSM(dataset=dataset, max_samples=len(dataset))
    adv_small = fgsm.attack(model, epsilon=0.1)
    adv_large = fgsm.attack(model, epsilon=0.3)

    assert not torch.allclose(adv_small, adv_large)

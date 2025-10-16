import pytest


torch = pytest.importorskip("torch")


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

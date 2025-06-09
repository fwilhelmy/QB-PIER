import pytest

# Skip tests if required libraries are missing
torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from src.utils import calculate_accuracy


def test_calculate_accuracy_all_correct():
    y_hat = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    y = torch.tensor([1, 0])
    acc = calculate_accuracy(y_hat, y)
    assert acc.item() == pytest.approx(1.0)


def test_calculate_accuracy_partial():
    y_hat = torch.tensor([[0.6, 0.4], [0.1, 0.9], [0.2, 0.8]])
    y = torch.tensor([0, 1, 1])
    acc = calculate_accuracy(y_hat, y)
    expected = 2 / 3
    assert acc.item() == pytest.approx(expected)

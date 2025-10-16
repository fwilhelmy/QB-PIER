import math

import pytest
import torch
from torch.utils.data import Dataset

from datasets import BaseDataset


class _TensorDataset(Dataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor) -> None:
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class _ConcreteDataset(BaseDataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor) -> None:
        self._data = data
        self._targets = targets
        super().__init__(train=True)
        self.mean = None
        self.std = None

    def load_dataset(self, train: bool, **kwargs):
        return _TensorDataset(self._data, self._targets)


@pytest.fixture
def simple_dataset():
    data = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    targets = torch.zeros(10, dtype=torch.long)
    return _ConcreteDataset(data, targets)


def test_calculate_split_sizes(simple_dataset):
    train_samples, eval_samples = simple_dataset.calculate_split_sizes(0.6, 0.2)
    assert train_samples == 6
    assert eval_samples == 2


def test_split_and_load_produces_expected_lengths(simple_dataset):
    train_loader, eval_loader = simple_dataset.split_and_load(
        train_samples_size=6,
        eval_samples_size=2,
        train_batch_size=4,
        eval_batch_size=4,
    )

    assert len(train_loader.dataset) == 6
    assert len(eval_loader.dataset) == 2
    assert train_loader.batch_size == 4
    assert eval_loader.batch_size == 2


def test_split_and_load_validates_total_requested_size(simple_dataset):
    with pytest.raises(ValueError):
        simple_dataset.split_and_load(
            train_samples_size=8,
            eval_samples_size=5,
            train_batch_size=4,
            eval_batch_size=4,
        )


def test_get_stats_calculates_values(monkeypatch):
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.zeros(2, dtype=torch.long)
    dataset = _ConcreteDataset(data, targets)

    base_module = __import__("datasets.BaseDataset", fromlist=["tqdm"])
    monkeypatch.setattr(base_module, "tqdm", lambda iterable, **kwargs: iterable)

    mean, std = dataset.get_stats()

    expected_mean = 2.5
    expected_std = math.sqrt(1.25)

    assert pytest.approx(mean) == expected_mean
    assert pytest.approx(std) == expected_std

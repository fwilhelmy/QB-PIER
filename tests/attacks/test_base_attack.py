from torch.utils.data import Dataset

from attacks import BaseAttack


class _DummyDataset(Dataset):
    def __init__(self, length: int) -> None:
        self._length = length

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._length

    def __getitem__(self, index):  # pragma: no cover - unused in tests
        raise NotImplementedError


class _ConcreteAttack(BaseAttack):
    def attack(self, model, epsilon: float, **kwargs):  # pragma: no cover - not exercised
        return model


def test_base_attack_defaults_to_dataset_length():
    dataset = _DummyDataset(length=10)
    attack = _ConcreteAttack(
        attack_name="dummy",
        dataset=dataset,
        max_samples=None,
    )

    assert attack.attack_name == "dummy"
    assert attack.dataset is dataset
    assert attack.max_samples == len(dataset)
    assert attack.batch_size == 64


def test_base_attack_respects_explicit_configuration():
    dataset = _DummyDataset(length=10)
    attack = _ConcreteAttack(
        attack_name="dummy",
        dataset=dataset,
        max_samples=5,
        batch_size=16,
    )

    assert attack.max_samples == 5
    assert attack.batch_size == 16

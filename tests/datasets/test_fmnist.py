import sys

from datasets import FMNIST


def test_fmnist_initializes_underlying_dataset(monkeypatch):
    module = sys.modules[FMNIST.__module__]

    monkeypatch.setattr(module, "ToTensor", lambda: object())

    class DummyFashionMNIST:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(module, "FashionMNIST", DummyFashionMNIST)

    dataset = FMNIST(root="/tmp/fmnist", train=False)

    assert isinstance(dataset.dataset, DummyFashionMNIST)

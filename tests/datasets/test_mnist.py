import sys

from datasets import MNIST


def test_mnist_initializes_underlying_dataset(monkeypatch):
    module = sys.modules[MNIST.__module__]

    monkeypatch.setattr(module, "ToTensor", lambda: object())

    class DummyMNIST:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(module, "TorchMNIST", DummyMNIST)

    dataset = MNIST(root="/tmp/mnist", train=True)

    assert isinstance(dataset.dataset, DummyMNIST)

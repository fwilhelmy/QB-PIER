from types import SimpleNamespace

import numpy as np

from datasets import PlusMinus


def test_plusminus_loads_tensor_dataset(monkeypatch):
    dummy = SimpleNamespace(
        img_train=np.zeros((1, 2, 2), dtype=np.float32),
        img_test=np.zeros((1, 2, 2), dtype=np.float32),
        labels_train=np.zeros((1,), dtype=np.int64),
        labels_test=np.zeros((1,), dtype=np.int64),
    )

    def fake_load(dataset_type, name):
        return [dummy]

    module = __import__(PlusMinus.__module__, fromlist=["qml"])
    monkeypatch.setattr(module.qml.data, "load", fake_load)

    dataset = PlusMinus(train=True)

    assert len(dataset.dataset) == 1

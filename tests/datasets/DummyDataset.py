import pytest


torch = pytest.importorskip("torch")


class DummyDataset(torch.utils.data.TensorDataset):
    def __init__(self, *, image_shape=(1, 28, 28), label=1, size=1):
        images = torch.zeros(size, *image_shape)
        labels = torch.full((size,), label, dtype=torch.long)
        super().__init__(images, labels)


class DummyDatasetManager:
    def __init__(self, dataset=None):
        self.dataset = dataset or DummyDataset()

    def compute_gradients(self, model, max_x=None, hot_reload: bool = False):
        return []

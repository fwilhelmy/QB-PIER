from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from data.DatasetManager import DatasetManager

class FMNISTManager(DatasetManager):
    def __init__(self, root: str, batch_size: int, num_workers: int):
        super().__init__(root, batch_size, num_workers)

    def load_dataset(self, train: bool):
        return FashionMNIST(root=self.root, train=train, download=True, transform=ToTensor())
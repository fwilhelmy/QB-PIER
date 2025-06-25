from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from src.data.DatasetManager import DatasetManager

class FMNISTManager(DatasetManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_dataset(self, train: bool):
        return FashionMNIST(root=self.root, train=train, download=True, transform=ToTensor())
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from data.DatasetManager import DatasetManager
from data.DatasetManagerFactory import DatasetManagerFactory


@DatasetManagerFactory.dataset_manager("fmnist")
class FMNISTManager(DatasetManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_dataset(self, train: bool):
        return FashionMNIST(root=self.root, train=train, download=True, transform=ToTensor())
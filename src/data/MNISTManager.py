from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from data.DatasetManager import DatasetManager
from data.DatasetManagerFactory import DatasetManagerFactory


@DatasetManagerFactory.dataset_manager("mnist")
class MNISTManager(DatasetManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Pre-calculated mean and std for the MNIST dataset
        self.mean, self.std = [0.1307], [0.3081]

    def load_dataset(self, train: bool):
        return MNIST(root=self.root, train=train, download=True, transform=ToTensor())
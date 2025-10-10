from torchvision.datasets import MNIST as TorchMNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from datasets import BaseDataset, DatasetFactory


@DatasetFactory.dataset("mnist")
class MNIST(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Pre-calculated mean and std for the MNIST dataset
        self.mean, self.std = [0.1307], [0.3081]

    def load_dataset(self, train: bool, **kwargs):
        return TorchMNIST(root=self.root, train=train, download=True, transform=ToTensor())
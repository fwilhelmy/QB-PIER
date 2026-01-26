from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from datasets import BaseDataset, DatasetFactory

@DatasetFactory.dataset("fmnist")
class FMNIST(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_dataset(self, train: bool, **kwargs):
        return FashionMNIST(root=self.root, train=train, download=True, transform=ToTensor())
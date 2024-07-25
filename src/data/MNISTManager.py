from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from data.DatasetManager import DatasetManager

class MNISTManager(DatasetManager):
    def __init__(self, root: str, batch_size: int, num_workers: int):
        # Pre-calculated mean and std for the MNIST dataset
        self.mean = [0.1307]
        self.std = [0.3081]

        super().__init__(root, batch_size, num_workers)

    def load_dataset(self, train: bool):
        return MNIST(root=self.root, train=train, download=True, transform=ToTensor())
    
        # If you want to apply normalization, use this instead
        # return MNIST(root=self.root, train=train, download=True, transform=Compose([
        #     ToTensor(),
        #     Normalize((self.mean,), (self.std,)),
        #     ]))
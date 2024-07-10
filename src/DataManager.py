from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split   

# TODO add batch_size and num_workers as parameters in the constructor
class DataManager:
    def __init__(self, dataset_name: str, root: str = "../data/processed"):
        self.dataset_name = dataset_name
        self.root = root
        self.dataset = self.load_dataset(train=True)

    def load_dataset(self, train: bool):
        if self.dataset_name == 'MNIST':
            return MNIST(root=self.root, train=train, download=True, transform=ToTensor())
        elif self.dataset_name == 'FMNIST':
            return FashionMNIST(root=self.root, train=train, download=True, transform=ToTensor())
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def calculate_split_sizes(self, train_split: float, val_split: float):
        dataset_length = len(self.dataset)
        train_samples = int(dataset_length * train_split)
        val_samples = int(dataset_length * val_split)
        return train_samples, val_samples

    def split_dataset(self, train_samples: int, val_samples: int, batch_size: int = 32):
        dataset_length = len(self.dataset)
        remaining_samples = dataset_length - (train_samples + val_samples)
        if remaining_samples < 0:
            raise ValueError("Sum of provided samples exceeds dataset length")
        
        # Split the dataset into training and validation sets
        split_lengths = [train_samples, val_samples, remaining_samples]
        train_dataset, val_dataset, _ = random_split(self.dataset, split_lengths)

        # Create data loaders for the sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

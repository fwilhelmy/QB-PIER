from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import random_split

def load_dataset(dataset_name, root="../data"):
    if dataset_name == 'MNIST':
        return MNIST(root=root, train=True, download=True, transform=ToTensor())
    elif dataset_name == 'FMNIST':
        return FashionMNIST(root=root, train=True, download=True, transform=ToTensor())
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_split_lengths(dataset_length, train_split, val_split, test_split, num_samples=None):
    if num_samples is None:
        train_samples = int(dataset_length * train_split)
        val_samples = int(dataset_length * val_split)
        test_samples = int(dataset_length * test_split)
    else:
        train_samples = num_samples
        val_samples = num_samples
        test_samples = num_samples

    remaining_samples = dataset_length - (train_samples + val_samples + test_samples)
    return [train_samples, val_samples, test_samples, remaining_samples]

def load_data(dataset_name, root="../data", train_split=0.4, val_split=0.4, test_split=0.2, num_samples=None):
    dataset = load_dataset(dataset_name, root)
    dataset_length = len(dataset)
    split_lengths = get_split_lengths(dataset_length, train_split, val_split, test_split, num_samples)
    train_dataset, val_dataset, test_dataset, _ = random_split(dataset, split_lengths)
    return train_dataset, val_dataset, test_dataset

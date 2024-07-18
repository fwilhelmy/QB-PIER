from Configuration import DATA_DIR

from data.FMNISTManager import FMNISTManager
from src.data.MNISTManager import MNISTManager

def DatasetManagerFactory(dataset_name: str, root: str = DATA_DIR, batch_size: int = 32, num_workers: int = 0):
    if dataset_name == 'MNIST':
        return MNISTManager(root=root, batch_size=batch_size, num_workers=num_workers)
    elif dataset_name == 'FMNIST':
        return FMNISTManager(root=root, batch_size=batch_size, num_workers=num_workers)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
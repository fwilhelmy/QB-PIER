from src.Configuration import DATA_DIR

from src.data.FMNISTManager import FMNISTManager
from src.data.MNISTManager import MNISTManager

def DatasetManagerFactory(dataset_name: str, root: str = DATA_DIR):
    if dataset_name == 'MNIST': return MNISTManager(root=root)
    elif dataset_name == 'FMNIST': return FMNISTManager(root=root)
    else: raise ValueError(f"Unsupported dataset: {dataset_name}")
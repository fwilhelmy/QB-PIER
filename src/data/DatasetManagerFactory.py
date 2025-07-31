from data.FMNISTManager import FMNISTManager
from data.MNISTManager import MNISTManager

DEFAULT_DATA_DIR = "./data/processed"

def DatasetManagerFactory(dataset_name: str, root: str = DEFAULT_DATA_DIR):
    dataset_name = dataset_name.lower()
    if dataset_name == 'mnist': return MNISTManager(root=root)
    elif dataset_name == 'fmnist': return FMNISTManager(root=root)
    else: raise ValueError(f"Unsupported dataset: {dataset_name}")

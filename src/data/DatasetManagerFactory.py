from data.FMNISTManager import FMNISTManager
from data.MNISTManager import MNISTManager
from data.PlusMinusManager import PlusMinusManager

DEFAULT_DATA_DIR = "./data/processed"

def DatasetManagerFactory(dataset_name: str, root: str = DEFAULT_DATA_DIR):
    if dataset_name == 'MNIST':
        return MNISTManager(root=root)
    elif dataset_name == 'FMNIST':
        return FMNISTManager(root=root)
    elif dataset_name.upper() in {"PLUSMINUS", "PLUS_MINUS"}:
        return PlusMinusManager(root=root)
    else: raise ValueError(f"Unsupported dataset: {dataset_name}")

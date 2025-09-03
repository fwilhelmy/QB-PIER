from data.FMNISTManager import FMNISTManager
from data.MNISTManager import MNISTManager
from data.PlusMinusManager import PlusMinusManager

def DatasetManagerFactory(dataset_name: str, root: str = "./data/processed"):
    dataset_name = dataset_name.lower()
    if dataset_name == 'mnist':
        return MNISTManager(root=root)
    elif dataset_name == 'fmnist':
        return FMNISTManager(root=root)
    elif dataset_name in "plusminus":
        return PlusMinusManager(root=root)
    else: raise ValueError(f"Unsupported dataset: {dataset_name}")

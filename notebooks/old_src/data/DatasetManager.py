from torch.utils.data import DataLoader, random_split
import torch
from tqdm import tqdm
import numpy as np

class DatasetManager:
    def __init__(self, root: str, batch_size: int, num_workers: int):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = self.load_dataset(train=True)

    def load_dataset(self, train: bool):
        raise NotImplementedError("This method should be overridden in subclasses")
    
    def __calculate_stats(self):
        psum = torch.tensor([0.0])
        psum_sq = torch.tensor([0.0])
        
        for data, _ in tqdm(self.dataset):
            psum += data.sum()
            psum_sq += (data ** 2).sum()
        
        count = np.prod(self.dataset.data.shape)
        
        self.mean = psum / count
        self.var = (psum_sq / count) - (self.mean ** 2)
        self.std = torch.sqrt(self.var)

    def mount_dataloader(self, dataset, shuffle: bool = False):
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            num_workers=self.num_workers,
            persistent_workers=(True if self.num_workers > 1 else False)) 
            # TODO Investigate if this is necessary to help in real long experiments
            
    def get_stats(self):
        if None in [self.mean, self.std]:
            self.__calculate_stats()
        
        return self.mean.item(), self.std.item()

    def calculate_split_sizes(self, train_split: float, val_split: float):
        dataset_length = len(self.dataset)
        train_samples = int(dataset_length * train_split)
        val_samples = int(dataset_length * val_split)
        return train_samples, val_samples

    def split_dataset(self, train_samples: int, val_samples: int):
        dataset_length = len(self.dataset)
        remaining_samples = dataset_length - (train_samples + val_samples)
        if remaining_samples < 0:
            raise ValueError("Sum of provided samples exceeds dataset length")
        
        # Split the dataset into training and validation sets
        split_lengths = [train_samples, val_samples, remaining_samples]
        train_dataset, val_dataset, _ = random_split(self.dataset, split_lengths)

        # Create data loaders for the sets
        train_loader = self.mount_dataloader(train_dataset, shuffle=True)
        val_loader = self.mount_dataloader(val_dataset, shuffle=False)

        return train_loader, val_loader

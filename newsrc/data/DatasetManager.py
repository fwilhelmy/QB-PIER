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
            
    def get_stats(self):
        if None in [self.mean, self.std]:
            self.__calculate_stats()
        
        return self.mean.item(), self.std.item()

    def calculate_split_sizes(self, train_split: float, eval_split: float):
        dataset_length = len(self.dataset)
        train_samples = int(dataset_length * train_split)
        eval_samples = int(dataset_length * eval_split)
        return train_samples, eval_samples

    def split_and_load(self, train_samples_size, eval_samples_size, train_batch_size, eval_batch_size, num_workers=0):
        dataset_length = len(self.dataset)
        remaining_samples = dataset_length - (train_samples_size + eval_samples_size)
        if remaining_samples < 0:
            raise ValueError("Sum of provided samples exceeds dataset length")
        
        # Split the dataset into training and evalidation sets
        split_lengths = [train_samples_size, eval_samples_size, remaining_samples]
        train_dataset, eval_dataset, _ = random_split(self.dataset, split_lengths)

        # Create data loaders for the sets
        train_loader = DataLoader(
            train_dataset, 
            batch_size=min(train_batch_size, len(train_dataset)),
            shuffle=True)
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=min(eval_batch_size, len(eval_dataset)),
            shuffle=False)

        return train_loader, eval_loader

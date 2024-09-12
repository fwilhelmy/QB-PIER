import torch
from DataManager import DataManager
from Configuration import LIGHTNING_DIR
from tqdm import tqdm

def calculate_dataset_stats(loader):
    psum = torch.tensor([0.0])
    psum_sq = torch.tensor([0.0])
    
    for data, _ in tqdm(loader):
        psum += data.sum()
        psum_sq += (data ** 2).sum()
    
    count = len(loader.dataset) * 28 * 28
    
    mean = psum / count
    var = (psum_sq / count) - (mean ** 2)
    std = torch.sqrt(var)
    
    return mean.item(), var.item(), std.item()

def calculate_batch_stats(loader):
    batch_means = []
    batch_stds = []
    
    for data, _ in tqdm(loader):
        batch_mean = data.mean()
        batch_std = data.std()
        batch_means.append(batch_mean.item())
        batch_stds.append(batch_std.item())
    
    return batch_means, batch_stds

data_manager = DataManager(dataset_name='MNIST')
train_split, val_split = data_manager.calculate_split_sizes(1,0)
train_loader, val_loader = data_manager.split_dataset(train_samples=train_split, val_samples=val_split, batch_size=64)

# Calculate stats for the whole dataset
total_mean, total_var, total_std = calculate_dataset_stats(train_loader)
print('Dataset stats:')
print('- mean: {:.4f}'.format(total_mean))
print('- var:  {:.4f}'.format(total_var))
print('- std:  {:.4f}'.format(total_std))

# Calculate stats per batch
batch_means, batch_stds = calculate_batch_stats(train_loader)
print('Batch stats:')
print('- mean of means: {}'.format(batch_means[0:5]))
print('- mean of stds: {}'.format(batch_stds[0:5]))

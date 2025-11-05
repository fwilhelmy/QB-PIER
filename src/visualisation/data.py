import logging
import random

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from lightning.pytorch.utilities import rank_zero_only

from LoggingService import LoggingService

log = logging.getLogger(__name__)

# Function to print general information about the dataset
@rank_zero_only
def print_dataset_info(dataset):
    print("Dataset Information:")
    print("Number of samples: %s", len(dataset))
    if hasattr(dataset, 'classes'):
        print("Number of classes: %s", len(dataset.classes))
        print("Classes: %s", dataset.classes)

# Function to plot class distribution histogram
@rank_zero_only
def plot_class_distribution(dataset, logdir="logs/visualisation"):
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset:
        class_counts[label] += 1

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(dataset.classes)), class_counts, color='blue', alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Class Distribution in Dataset')
    plt.xticks(range(len(dataset.classes)), dataset.classes)

    # Saving the results
    LoggingService().save_figure(plt.gcf(), "class_distribution", logdir)

# Function to plot class visualizations: sample, mean, and standard deviation images
@rank_zero_only
def plot_class_visualizations(dataset, logdir="logs/visualisation"):
    mean_images = []
    std_images = []
    sample_images = []

    for i in range(len(dataset.classes)):
        class_images = torch.stack([img for img, label in dataset if label == i])
        mean_images.append(class_images.mean(dim=0))
        std_images.append(class_images.std(dim=0))
        sample_images.append(class_images[random.randint(0, len(class_images) - 1)])

    fig, axes = plt.subplots(4, len(dataset.classes), figsize=(15, 8))
    for i in range(len(dataset.classes)):
        # Top row for class labels
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Class {dataset.classes[i]}")
        
        # Sample image
        axes[1, i].imshow(sample_images[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Sample')
        
        # Mean image
        axes[2, i].imshow(mean_images[i].squeeze(), cmap='gray')
        axes[2, i].axis('off')
        axes[2, i].set_title('Mean')
        
        # STD image
        axes[3, i].imshow(std_images[i].squeeze(), cmap='gray')
        axes[3, i].axis('off')
        axes[3, i].set_title('STD')
        
    plt.tight_layout()

    # Saving the results
    LoggingService().save_figure(fig, "class_visualizations", logdir)

# Function to perform t-SNE and plot the results
@rank_zero_only
def plot_tsne(dataset, subset_size=1000, logdir="logs/visualisation"):
    indices = random.sample(range(len(dataset)), subset_size)
    subset_data = [dataset[i] for i in indices]
    images, labels = zip(*subset_data)
    images = torch.stack([img.view(-1) for img in images])
    # scikit-learn expects array-like (e.g. numpy), convert from torch.Tensor
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(images)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(len(dataset.classes)))
    plt.title('t-SNE of Dataset')

    # Saving the results
    LoggingService().save_figure(plt.gcf(), "tsne", logdir)

# Function to plot sample images from the dataset
@rank_zero_only
def plot_sample_images(dataset, x=10, rows=2, cols=5, logdir="logs/visualisation"):
    indices = random.sample(range(len(dataset)), x)
    images, labels = zip(*[dataset[i] for i in indices])

    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
    for i, (img, label) in enumerate(zip(images, labels)):
        ax = axes[i // cols, i % cols]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.tight_layout()

    # Saving the results
    LoggingService().save_figure(fig, "sample_images", logdir)
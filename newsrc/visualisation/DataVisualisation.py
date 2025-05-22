import torch
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
from LoggingService import LoggingService

# Function to print general information about the dataset
def print_dataset_info(dataset):
    print("Dataset Information:")
    print(f"Number of samples: {len(dataset)}")
    if hasattr(dataset, 'classes'):
        print(f"Number of classes: {len(dataset.classes)}")
        print(f"Classes: {dataset.classes}")

# Function to plot class distribution histogram
def plot_class_distribution(dataset):
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
    LoggingService().save_figure(plt, "class_distribution")

# Function to plot class visualizations: sample, mean, and standard deviation images
def plot_class_visualizations(dataset):
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
    LoggingService().save_figure(plt, "class_visualizations")

# Function to perform t-SNE and plot the results
def plot_tsne(dataset, subset_size=1000):
    indices = random.sample(range(len(dataset)), subset_size)
    subset_data = [dataset[i] for i in indices]
    images, labels = zip(*subset_data)
    images = torch.stack([img.view(-1) for img in images])

    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(images)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(len(dataset.classes)))
    plt.title('t-SNE of Dataset')

    # Saving the results
    LoggingService().save_figure(plt, "tsne")

# Function to plot sample images from the dataset
def plot_sample_images(dataset, x=10, rows=2, cols=5):
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
    LoggingService().save_figure(plt, "sample_images")
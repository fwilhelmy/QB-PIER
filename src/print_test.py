from LoggingService import LoggingService
from datasets import MNIST, FMNIST, PlusMinus
from visualisation.data import (
    print_dataset_info,
    plot_class_distribution,
    plot_class_visualizations,
    plot_tsne,
    plot_sample_images,
)
    
if __name__ == "__main__":
    logdir = "logs/vis_test/plusminus"
    LoggingService(logdir)
    mnist = PlusMinus(train=True, download=True)
    print_dataset_info(mnist.dataset)
    plot_class_distribution(mnist.dataset)
    plot_class_visualizations(mnist.dataset)
    plot_tsne(mnist.dataset, subset_size=500)
    plot_sample_images(mnist.dataset, x=10, rows=2, cols=5)
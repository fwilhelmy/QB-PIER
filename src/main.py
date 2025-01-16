from models.Classical.LeNet import LeNet
from data.DatasetManagerFactory import DatasetManagerFactory
from visualisation.AttacksVisualisation import plot_adversarial_examples
import matplotlib.pyplot as plt
from LoggingService import LoggingService
from attacks.FGSMAttack import FGSMAttack
from utils import save_figure
from models.Hybrid.HQNN_Parallel import HQNN_Parallel
from utils import calculate_accuracy
import numpy as np
import torch
from torch.utils.data import DataLoader

def visualize_attacks_results(model_name, adv_exemples, num_images=5):
    # Sort the epsilon levels for consistent ordering
    epsilons = sorted(adv_exemples.keys())
    num_epsilons = len(epsilons)

    # Set up the matplotlib figure
    fig, axes = plt.subplots(num_epsilons, num_images, figsize=(num_images * 2, num_epsilons * 2))
    fig.suptitle('Adversarial Attacks Visualization', fontsize=16)

    # If there's only one epsilon level or one image, axes may not be a 2D array
    if num_epsilons == 1:
        axes = [axes]
    else:
        axes = axes

    for row, eps in enumerate(epsilons):
        perturbed_images = adv_exemples[eps]
        for col in range(num_images):
            ax = axes[row][col] if num_epsilons > 1 else axes[col]
            
            # Ensure the index does not exceed the available images
            if col < len(perturbed_images):
                img = perturbed_images[col]
                # If the image has more than 2 dimensions, assume it's multi-channel and take the first channel
                if img.ndim == 3 and img.shape[0] in [1, 3]:
                    img = img.squeeze(0)
                elif img.ndim == 2:
                    img = img  # Grayscale image

                ax.imshow(img, cmap='gray')
            else:
                ax.axis('off')  # Hide axes if no image is available

            ax.axis('off')  # Hide axis ticks

            # Set titles for the first row
            if row == 0:
                ax.set_title(f"Image {col+1}", fontsize=10)

        # Set y-axis labels for each row
        axes[row][0].set_ylabel(f"Epsilon: {eps}", rotation=0, labelpad=50, fontsize=12, va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
    
    plt.show()
    save_figure("adv_atcks", plt, model_name)
    plt.close()

if __name__ == "__main__":
    # Create services instance
    logs_manager = LoggingService()

    # Create results lists
    training_results = {}
    evaluation_results = {}

    # Load the dataset using DataManager
    data_manager = DatasetManagerFactory('MNIST')
    train_split, val_split = data_manager.calculate_split_sizes(1, 0)
    train_dataset, val_dataset = data_manager.split_dataset(train_samples=1500, val_samples=300, batch_size=64)
    
    # Training models
    # TODO Do quantum models work?
    # model1 = HQNN_Parallel(model_name='HQNN_Parallel_Test_1', max_epochs=8, in_channels=1, classes=10)
    # model1.fit(train_dataset, val_dataset)
    # training_results[model1.model_name] = model1.trainer.logged_metrics

    model2 = LeNet(model_name='LeNet_Test_2', in_channels=1, classes=10)
    model2.fit(train_dataset, val_dataset)
    training_results[model2.model_name] = model2.trainer.logged_metrics
    
    models = [model2]

    # Performing attacks
    attacks = [FGSMAttack(x) for x in [0, 0.3, 1.0]]
    test_dataset = data_manager.load_dataset(train=False)
    
    for model in models:
        evaluation_results = {}
        for attack in attacks:
            adv_examples = attack.evaluate_attack(model, test_dataset)
            evaluation_results[attack.epsilon] = adv_examples
            # TODO Figure out how to use the datasets to do predictions now that I have them
            #og_pred = model(test_dataset)
            #adv_pred = model(adv_examples)
        visualize_attacks_results(model.model_name, evaluation_results)

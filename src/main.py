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

def plot_attack_effect_on_accuracy(models, epsilons, accuracies):
    plt.figure(figsize=(10, 6))

    for model in models:
        plt.plot(epsilons, accuracies[model.model_name], label=model.model_name, marker='o')

    plt.xlabel('Attack Strength (Epsilon)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Effect of Adversarial Attacks on Model Accuracy', fontsize=14)
    plt.xticks(epsilons, [f"{e:.2f}" for e in epsilons], rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Models', fontsize=10)
    plt.tight_layout()

    plt.show()
    save_figure("accs", plt)
    plt.close()

if __name__ == "__main__":
    # Create services instance
    logs_manager = LoggingService()

    # Create results lists
    training_results = {}
    evaluation_results = {}

    # Load the dataset using DataManager
    data_manager = DatasetManagerFactory('MNIST')
    train_split, val_split = data_manager.calculate_split_sizes(0.7, 0.3)
    train_dataset, val_dataset = data_manager.split_dataset(train_samples=train_split, val_samples=val_split, batch_size=64)
    
    # Configuration of models
    models = []

    models.append(HQNN_Parallel(model_name='HQNN_Parallel_8', max_epochs=8, in_channels=1, classes=10))
    models.append(HQNN_Parallel(model_name='HQNN_Parallel_24', max_epochs=24, in_channels=1, classes=10))
    models.append(LeNet(model_name='LeNet_8', max_epochs=8, in_channels=1, classes=10))
    models.append(LeNet(model_name='LeNet_24', max_epochs=24, in_channels=1, classes=10))

    for model in models:
        model.fit(train_dataset, val_dataset)
        training_results[model.model_name] = model.trainer.logged_metrics

    # Performing attacks
    test_dataset = data_manager.load_dataset(train=False)
    epsilons = [x/100 for x in range(0, 105, 5)]
    fgsm = FGSMAttack(test_dataset)
    
    accuracies = {}
    for model in models:
        evaluation_results = {}
        accuracies[model.model_name] = []
        for epsilon in epsilons:
            adv_examples = fgsm.attack(model, epsilon)
            evaluation_results[epsilon] = adv_examples
            # TODO Figure out how to use the datasets to do predictions now that I have them
            og_preds = model(test_dataset.data.unsqueeze(1).to(torch.float))
            og_acc = calculate_accuracy(og_preds, test_dataset.targets)
            adv_pred = model(torch.stack(adv_examples))
            adv_acc = calculate_accuracy(adv_pred, test_dataset.targets[0:len(adv_examples)])
            accuracies[model.model_name].append(adv_acc)
            print("og_acc", og_acc)
            print("adv_acc", adv_acc)
        visualize_attacks_results(model.model_name, evaluation_results)

    plot_attack_effect_on_accuracy(models, epsilons, accuracies)
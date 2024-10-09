import torch
from torch.utils.data import DataLoader

from models.Classical.LeNet import LeNet
from ModelsManager import ModelsManager, ExampleConfiguration
from data.DatasetManagerFactory import DatasetManagerFactory
from visualisation.AttacksVisualisation import plot_adversarial_examples
import matplotlib.pyplot as plt
from models.ModelConfiguration import ModelConfiguration
import ModelsManager
import AttacksManager
import src.LoggingService as LoggingService

def plot_adversarial_examples(results):
    fig, axes = plt.subplots(len(results['epsilons']), 4, figsize=(12, 12))
    fig.suptitle('Original and Adversarial Images for Different Epsilons')

    for i, (epsilon, examples) in enumerate(zip(results['epsilons'], results['examples'])):
        exp1_original_img, exp2_original_img = examples['original_images'][0:2]
        exp1_perturbed_img, exp2_perturbed_img = examples['perturbed_images'][0:2]
        exp1_original_pred, exp2_original_pred = examples['original_preds'][0:2]
        exp1_perturbed_pred, exp2_perturbed_pred = examples['perturbed_preds'][0:2]

        exp1_original_pred = exp1_original_pred.argmax().item()
        exp1_perturbed_pred = exp1_perturbed_pred.argmax().item()
        exp2_original_pred = exp2_original_pred.argmax().item()
        exp2_perturbed_pred = exp2_perturbed_pred.argmax().item()

        # Plot original image 1
        axes[i, 0].imshow(exp1_original_img.squeeze(), cmap='gray')
        axes[i, 0].axis('off')

        # Plot adversarial image 1
        axes[i, 1].imshow(exp1_perturbed_img.squeeze(), cmap='gray')
        axes[i, 1].axis('off')

        # Plot original image 2
        axes[i, 2].imshow(exp2_original_img.squeeze(), cmap='gray')
        axes[i, 2].axis('off')

        # Plot adversarial image 2
        axes[i, 3].imshow(exp2_perturbed_img.squeeze(), cmap='gray')
        axes[i, 3].axis('off')

        # Set y-axis title to epsilon value
        axes[i, 0].set_ylabel(f'Epsilon: {epsilon}', rotation=90, size='large')

    axes[0, 0].set_title(f'Img1 Org [{exp1_original_pred}]')
    axes[0, 1].set_title(f'Img1 Pertub [{exp1_perturbed_pred}]')
    axes[0, 2].set_title(f'Img2 Org [{exp2_original_pred}]')
    axes[0, 3].set_title(f'Img2 Pertub [{exp2_perturbed_pred}]')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Saving the results to the ./results directory
    plt.savefig(f"./results/adversarial_examples.png")
    plt.show()

if __name__ == "__main__":
    # Create services instance
    logs_manager = LoggingService()
    attack_manager = AttacksManager()

    # Create results lists
    training_results = {}
    evaluation_results = {}

    # Load the dataset using DataManager
    data_manager = DatasetManagerFactory('MNIST')
    train_split, val_split = data_manager.calculate_split_sizes(1, 0)
    train_dataset, val_dataset = data_manager.split_dataset(train_samples=500, val_samples=150, batch_size=64)
    
    # Training models
    model1 = LeNet('LeNet_Test_1', numChannels=1, classes=10)
    model1.trainer.fit(model1, train_dataset, val_dataset)
    training_results[model1.model_name] = model1.trainer.logged_metrics

    model2 = LeNet('LeNet_Test_2', numChannels=1, classes=10)
    model2.trainer.fit(model2, train_dataset, val_dataset)
    training_results[model2.model_name] = model2.trainer.logged_metrics
    
    models = [model1, model2]

    # Performing attacks
    attack_manager.register_attack(f'FGSM', 0)
    attack_manager.register_attack(f'FGSM', 0.5)
    attack_manager.register_attack(f'FGSM', 1)

    # Perform attacks
    attack_results = model_manager.batch_attack(test_dataset)

    # Save results
    # model_manager.batch_save(training_results, evaluation_results, None)

    # TODO DOESNT STORE CORRECTLY THE ATTACK TYPE
    plot_adversarial_examples(attack_results['LeNet_Test_2'])
            

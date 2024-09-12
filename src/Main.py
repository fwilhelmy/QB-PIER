import torch
from torch.utils.data import DataLoader

from models.Classical.LeNet import LeNet
from ModelsManager import ModelsManager, ExampleConfiguration
from data.DatasetManagerFactory import DatasetManagerFactory
from visualisation.AttacksVisualisation import plot_adversarial_examples
import matplotlib.pyplot as plt

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
    # Load the dataset using DataManager
    data_manager = DatasetManagerFactory('MNIST')
    train_split, val_split = data_manager.calculate_split_sizes(1, 0)
    train_loader, val_loader = data_manager.split_dataset(train_samples=500, val_samples=150, batch_size=64)

    # Define model configuration
    config = ExampleConfiguration(
        optimiser=torch.optim.Adam,
        init_lr=0.001,
        loss_function=torch.nn.CrossEntropyLoss(),
        max_epochs=6,
        batch_size=64,
        gpus=1  # Set to None for CPU training
    )

    # Create ModelManager instance
    model_manager = ModelsManager(10)

    # Register models
    model1 = LeNet(config, numChannels=1, classes=10)
    # model_manager.register_model('LeNet_Test_1', model1, config)

    model2 = LeNet(config, numChannels=1, classes=10)
    model_manager.register_model('LeNet_Test_2', model2, config)

    # Register attacks
    epsilons = [0, .15, .6, 0.9, 1.2, 5, 10]
    for eps in epsilons:
        model_manager.register_attack(f'FGSM', epsilon=eps)

    # Train models
    training_results = model_manager.batch_train(train_loader, val_loader)

    # Evaluate models
    test_dataset = data_manager.load_dataset(train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    evaluation_results = model_manager.batch_evaluate(test_loader)

    # Perform attacks
    attack_results = model_manager.batch_attack(test_dataset)

    # Save results
    # model_manager.batch_save(training_results, evaluation_results, None)

    # TODO DOESNT STORE CORRECTLY THE ATTACK TYPE
    plot_adversarial_examples(attack_results['LeNet_Test_2'])
            

import torch
from torch.utils.data import DataLoader

from models.Classical.LeNet import LeNet
from attacks.FGSM import evaluate_attack
from ModelManager import ModelManager, ExampleConfiguration
from data.DatasetManagerFactory import DatasetManagerFactory


if __name__ == "__main__":
    # Load the dataset using DataManager
    data_manager = DatasetManagerFactory('MNIST')
    train_split, val_split = data_manager.calculate_split_sizes(1, 0)
    train_loader, val_loader = data_manager.split_dataset(train_samples=train_split, val_samples=val_split, batch_size=64)

    # Define model configuration
    config = ExampleConfiguration(
        optimiser=torch.optim.Adam,
        init_lr=0.001,
        loss_function=torch.nn.CrossEntropyLoss(),
        max_epochs=10,
        batch_size=64,
        gpus=1  # Set to None for CPU training
    )

    # Create ModelManager instance
    model_manager = ModelManager(10)

    # Register models
    model1 = LeNet(config, numChannels=1, classes=10)
    model_manager.register_model('LeNet_Test_1', model1, config)

    model2 = LeNet(config, numChannels=1, classes=10)
    model_manager.register_model('LeNet_Test_2', model2, config)

    # Register attacks
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    for eps in epsilons:
        model_manager.register_attack(f'fgsm({eps})', evaluate_attack, epsilon=eps)

    # Train models
    training_results = model_manager.batch_train(train_loader, val_loader)

    # Evaluate models
    test_dataset = data_manager.load_dataset(train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    evaluation_results = model_manager.batch_evaluate(test_loader)

    # Perform attacks
    # attack_results = model_manager.batch_attack(test_loader)

    # Save results
    model_manager.batch_save(training_results, evaluation_results, None)

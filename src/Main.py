import logging
from datetime import datetime
import os
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from models.BaseModel import BaseModel
from models.Classical.LeNet import LeNet
from models.Hybrid.HQNN_Quanv import HQNN_Quanv
from models.Hybrid.HQNN_Parallel import HQNN_Parallel

from attacks.FGSM import evaluate_attack

from visualisation.AttacksVisualisation import plot_accuracy_vs_epsilon
from visualisation.TrainingVisualisation import plot_training_result, plot_models_together

from DataManager import DataManager
from Configuration import LIGHTNING_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Arrays to store registered models and attacks
models = []
attacks = []

# TODO MAX EPOCHS AND STUFF SHOULD BE IN A CONFIG
trainer = Trainer(max_epochs=4, default_root_dir=LIGHTNING_DIR)

# Default value for configuration is Configuration()
def register_model(model_name, model, configuration):
    models.append({'name': model_name, 'model': model, 'configuration': configuration, 'trainer': trainer})
    logging.info(f"Model '{model_name}' registered.")
    return models

def register_attack(attack_name, attack_function, epsilon):
    attacks.append({'name': attack_name, 'function': attack_function, 'epsilon': epsilon})
    logging.info(f"Attack '{attack_name}' registered with epsilon {epsilon}.")
    return attacks

def batch_train(models, train_loader, val_loader):
    results = {}
    for model in models:
        model_name = model['name']
        logging.info(f"Training model '{model_name}'...")
        trainer = model['trainer']
        model_instance = model['model']
        trainer.fit(model_instance, train_loader, val_loader)
        results[model_name] = trainer.logged_metrics
    return results

def batch_evaluate(models, test_loader):
    results = {}
    for model in models:
        model_name = model['name']
        logging.info(f"Evaluating model '{model_name}'...")
        trainer = model['trainer']
        model_instance = model['model']
        results[model_name] = trainer.test(model_instance, dataloaders=test_loader)[0]
    return results

def batch_attack(models, attacks, test_loader):
    results = {}
    for model in models:
        model_name = model['name']
        results[model_name] = {'epsilons': [], 'accuracies': [], 'examples': []}
        for attack in attacks:
            attack_name = attack['name']
            attack_function = attack['function']
            model_instance = model['model']
            epsilon = attack['epsilon']
            logging.info(f"Performing '{attack_name}' on model '{model_name}'...")
            acc, adv_examples = attack_function(model_instance, test_loader, epsilon)
            results[model_name]['epsilons'].append(epsilon)
            results[model_name]['accuracies'].append(acc)
            results[model_name]['examples'].append(adv_examples)
    return results

def batch_save(models, training_results, evaluation_results, attack_results):
    """Visualize the training results and models together."""

    # Create a directory to store the results
    timestring = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
    path = f"results/{timestring}"
    os.makedirs(path, exist_ok=True)

    for model in models:
        model_name = model['name']
        model_instance = model['model']
        model_path = f"{path}/{model_name}"
        os.makedirs(model_path)

        # Save the model
        model_file = f"{model_path}/model.pt"
        torch.save(model_instance.state_dict(), model_file)

        # Plot the training results
        plot_training_result(path, model, training_results[model_name])

        # Plot results for each attack
        if attack_results is not None:
            for attack_name, result in attack_results[model_name].items():
                plot_accuracy_vs_epsilon(path, model_name, attack_name, result['accuracies'], result['epsilons'])

    plot_models_together(path, training_results)

# Example configuration
class ExampleConfiguration:
    def __init__(self, optimiser, init_lr, loss_function, max_epochs=10, batch_size=64, gpus=None):
        self.optimiser = optimiser
        self.init_lr = init_lr
        self.loss_function = loss_function
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.gpus = gpus


# Load the dataset using DataManager
data_manager = DataManager(dataset_name='MNIST')
train_loader, val_loader = data_manager.split_dataset(train_samples=500, val_samples=100, batch_size=64)

# Define model configuration
config = ExampleConfiguration(
    optimiser=torch.optim.Adam,
    init_lr=0.001,
    loss_function=torch.nn.CrossEntropyLoss(),
    max_epochs=10,
    batch_size=64,
    gpus=1  # Set to None for CPU training
)

# Register models
model1 = LeNet(config, numChannels=1, classes=10)
models = register_model('LeNet_Test_1', model1, config)

model2 = LeNet(config, numChannels=1, classes=10)
models = register_model('LeNet_Test_2', model2, config)

# Register attacks
epsilons = [0, .05, .1, .15, .2, .25, .3]
for eps in epsilons:
    register_attack(f'fgsm({eps})', evaluate_attack, epsilon=eps)

# Train models
training_results = batch_train(models, train_loader, val_loader)

# Evaluate models
test_dataset = data_manager.load_dataset(train=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
evaluation_results = batch_evaluate(models, test_loader)

# Perform attacks
# attack_results = batch_attack(models, attacks, test_loader)

# Save results
batch_save(models, training_results, evaluation_results, None)

import logging
from ModelTrainer import ModelTrainer
from datetime import datetime
import os
import torch

from src.models.Classical.LeNet import LeNet
from models.Hybrid.HQNN_Quanv import HQNN_Quanv
from models.Hybrid.HQNN_Parallel import HQNN_Parallel

from attacks.FGSM import fgsm_attack

from Visualisation import plot_training_result, plot_models_together
from DataLoader import load_dataset, split_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dictionary to store registered models and attacks
models = {}
attacks = {}

def register_model(model_name, model_trainer):
    """Register a model with its configuration."""
    if model_name not in models:
        models[model_name] = {'trainer': model_trainer}
        logging.info(f"Model '{model_name}' registered.")
    else:
        logging.warning(f"Model '{model_name}' already registered.")
    return models

def register_attack(attack_name, attack_function):
    """Register an attack function to the pipeline."""
    if attack_name not in attacks:
        attacks[attack_name] = attack_function
        logging.info(f"Attack '{attack_name}' registered.")
    else:
        logging.warning(f"Attack '{attack_name}' already registered.")
    return attacks

def batch_train(models, train_dataset, val_dataset):
    """Train all registered models."""
    results = {}
    for model_name, model_info in models.items():
        logging.info(f"Training model '{model_name}'...")
        trainer = model_info['trainer']
        results[model_name] = trainer.train(train_dataset, val_dataset)
        plot_training_result(rootdir, model_name, trainer, results[model_name])

    return results

def batch_evaluate(models, test_dataset):
    """Evaluate all registered models."""
    results = {}
    for model_name, model_info in models.items():
        logging.info(f"Evaluating model '{model_name}'...")
        trainer = model_info['trainer']
        results[model_name] = trainer.evaluate(test_dataset)
    return results

def batch_attack(models, attack_name, test_loader, epsilon):
    """Perform attacks on all registered models using the specified attack."""
    if attack_name not in attacks:
        logging.error(f"Attack '{attack_name}' is not registered.")
        return None

    attack_function = attacks[attack_name]
    results = {}
    for model_name, model_info in models.items():
        logging.info(f"Performing '{attack_name}' on model '{model_name}'...")
        model = model_info['trainer'].model
        acc, adv_examples = attack_function(model, test_loader, epsilon)
        results[model_name] = {
            'accuracy': acc,
            'adv_examples': adv_examples
        }
    return results

# Create a directory to store the results
rootdir = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
os.makedirs(f"results/{rootdir}", exist_ok=True)

dataset = load_dataset(dataset_name='MNIST', root="../data")
trainDataset, valDataset, testDataset = split_dataset(num_samples=128)

model = LeNet(numChannels=1, classes=10)
models = register_model('Model1_Test_64', ModelTrainer(model))
models = register_model('Model2_Test_70', ModelTrainer(model, batch_size=70))

attacks = register_attack('fgsm', fgsm_attack)

training_results = batch_train(models, trainDataset, valDataset)
evaluation_results = batch_evaluate(models, testDataset)
attack_results = batch_attack(models, 'fgsm', testDataset, epsilon=0.3)

plot_models_together(rootdir, training_results)
import logging
from ModelTrainer import ModelTrainer
from datetime import datetime
import os
import torch

from models.Classical.LeNet import LeNet
from models.Hybrid.HQNN_Quanv import HQNN_Quanv
from models.Hybrid.HQNN_Parallel import HQNN_Parallel

from attacks.FGSM import fgsm_attack

from visualisation.AttacksVisualisation import plot_accuracy_vs_epsilon
from visualisation.TrainingVisualisation import plot_training_result, plot_models_together

from DataLoader import load_dataset, split_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dictionary to store registered models and attacks
models = {}
attacks = {}

def register_model(model_name, model_trainer):
    models[model_name] = {'trainer': model_trainer}
    logging.info(f"Model '{model_name}' registered.")
    return models

def register_attack(attack_name, attack_function, epsilon):
    attacks[attack_name] = {'function': attack_function, 'epsilon': epsilon}
    logging.info(f"Attack '{attack_name}' registered with epsilon {epsilon}.")
    return attacks

def batch_train(models, train_dataset, val_dataset):
    results = {}
    for model_name, model_info in models.items():
        logging.info(f"Training model '{model_name}'...")
        trainer = model_info['trainer']
        results[model_name] = trainer.train(train_dataset, val_dataset)
    return results

def batch_evaluate(models, test_dataset):
    results = {}
    for model_name, model_info in models.items():
        logging.info(f"Evaluating model '{model_name}'...")
        trainer = model_info['trainer']
        results[model_name] = trainer.evaluate(test_dataset)
    return results

def batch_attack(models, attacks, test_dataset):
    results = {}
    for model_name, model_info in models.items():
        results[model_name] = {'epsilons': [], 'accuracies': [], 'examples': []}
        for attack_name, attack_info in attacks.items():
            attack_function = attack_info['function']
            model = model_info['trainer'].model
            epsilon = attack_info['epsilon']
            logging.info(f"Performing '{attack_name}' on model '{model_name}'...")
            acc, adv_examples = attack_function(model, test_dataset, epsilon)
            results[model_name]['epsilons'].append(epsilon)
            results[model_name]['accuracies'].append(acc)
            results[model_name]['examples'].append(adv_examples)
    return results

def batch_save(models, training_results, evaluation_results, attack_results):
    """Visualise the training results and models together."""

    # Create a directory to store the results
    timestring = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
    path = f"results/{timestring}"   
    os.makedirs(path, exist_ok=True)

    for model_name, model in models.items():
        model_path = f"{path}/{model_name}"
        os.makedirs(model_path)

        # Save the model
        model_file = f"{model_path}/model.pt"
        with open(model_file, 'wb') as f:
            torch.save(model.state_dict(), f)

        # Plot the training results
        plot_training_result(path, model_name, model, training_results[model_name])

        # Plot results for each attack
        for attack_name, result in attack_results[model_name].items():
            plot_accuracy_vs_epsilon(path, model_name, attack_name, result['accuracies'], result['epsilons'])

    plot_models_together(path, training_results)

dataset = load_dataset(dataset_name='MNIST', root="../data")
trainDataset, valDataset, testDataset = split_dataset(dataset, num_samples=128)

model = LeNet(numChannels=1, classes=10)
models = register_model('Model1_Test_64', ModelTrainer(model))
models = register_model('Model2_Test_70', ModelTrainer(model, batch_size=70))

epsilons = [0, .05, .1, .15, .2, .25, .3]
for eps in epsilons:
    register_attack(f'fgsm({eps})', fgsm_attack, epsilon=eps)

training_results = batch_train(models, trainDataset, valDataset)
evaluation_results = batch_evaluate(models, testDataset)
attack_results = batch_attack(models, attacks, testDataset)

batch_save(models, training_results, evaluation_results, attack_results)

from models.Classical.LeNet import LeNet
from data.DatasetManagerFactory import DatasetManagerFactory
from visualisation.AttacksVisualisation import plot_attack_effect_on_accuracy, visualize_attacks_results
from LoggingService import LoggingService
from attacks.FGSMAttack import FGSMAttack
from utils import save_figure
from models.Hybrid.HQNN_Parallel import HQNN_Parallel
from models.Hybrid.HQNN_Quanv import HQNN_Quanv
from utils import calculate_accuracy
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
from argparse import ArgumentParser
import time

def main(hparams):
    # Create services instance
    logs_manager = LoggingService()

    # Create results lists
    training_results = {}
    evaluation_results = {}

    # Load the dataset using DataManager
    print("workers : ", hparams.workers)
    data_manager = DatasetManagerFactory('MNIST', num_workers=hparams.workers)
    train_split, val_split = data_manager.calculate_split_sizes(0.7, 0.3)
    #train_dataset, val_dataset = data_manager.split_dataset(train_samples=train_split, val_samples=val_split)
    train_dataset, val_dataset = data_manager.split_dataset(train_samples=1500, val_samples=700)
    
    # Configuration of models
    models = []

    models.append(LeNet(model_name='LeNet_8', max_epochs=8, in_channels=1, classes=10))
    models.append(HQNN_Quanv(model_name='HQNN_Quanv_8', max_epochs=8, in_channels=1, classes=10))
    models.append(HQNN_Parallel(model_name='HQNN_Parallel_8', max_epochs=8, in_channels=1, classes=10))
    #models.append(HQNN_Parallel(model_name='HQNN_Parallel_24', max_epochs=24, in_channels=1, classes=10))
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
        og_preds = model(test_dataset.data.unsqueeze(1).to(torch.float))
        og_acc = calculate_accuracy(og_preds, test_dataset.targets)
        print("Accuracy of model : ", og_acc)
        for epsilon in epsilons:
            adv_examples = fgsm.attack(model, epsilon)
            evaluation_results[epsilon] = adv_examples
            adv_pred = model(torch.stack(adv_examples))
            adv_acc = calculate_accuracy(adv_pred, test_dataset.targets[:len(adv_examples)])
            accuracies[model.model_name].append(adv_acc)
            print(f"Accuracy of model {model.model_name} with epsilon {epsilon} : {adv_acc}")
        visualize_attacks_results(model.model_name, evaluation_results)
    plot_attack_effect_on_accuracy(models, epsilons, accuracies)

# Implement the parser (Make my own parser)
if __name__ == "__main__":
    # TODO add helpers for the ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-d", "--devices", type=int, default=1, help='TODO')
    parser.add_argument("-w", "--workers", type=int, default=0, help='TODO')
    args = parser.parse_args()
    start = time.time()
    main(args)
    end = time.time()
    print("Execution time: ", end - start)

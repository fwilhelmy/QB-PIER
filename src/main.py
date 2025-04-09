from models.Classical.LeNet import LeNet
from data.DatasetManagerFactory import DatasetManagerFactory
from visualisation.AttacksVisualisation import plot_attack_effect_on_accuracy, visualize_attacks_results
from visualisation.TrainingVisualisation import plot_training_result, plot_models_together
from LoggingService import LoggingService
from attacks.FGSMAttack import FGSMAttack
from attacks.PGDAttack import PGDAttack
from models.Hybrid.HQNN_Parallel import HQNN_Parallel
from models.Hybrid.HQNN_Quanv import HQNN_Quanv
from utils import calculate_accuracy
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.Quantum.PennyClassifier import PennyClassifier
import sys
from argparse import ArgumentParser
import time

def main(hparams):
    # Load the dataset using DataManager
    print("workers : ", hparams.workers)
    data_manager = DatasetManagerFactory('MNIST', num_workers=hparams.workers, batch_size=80)
    train_split, val_split = data_manager.calculate_split_sizes(0.7, 0.3)
    #train_dataset, val_dataset = data_manager.split_dataset(train_samples=train_split, val_samples=val_split)
    train_dataset, val_dataset = data_manager.split_dataset(train_samples=1500, val_samples=500)
    
    # Configuration of models
    models = []
    
    models.append(LeNet(model_name='LeNet_8', devices=hparams.devices, max_epochs=8, in_channels=1, classes=10))
    #models.append(LeNet(model_name='LeNet_32', devices=hparams.devices, max_epochs=32, in_channels=1, classes=10))
    models.append(HQNN_Parallel(model_name='HQNN_Parallel_8', devices=hparams.devices, max_epochs=8, in_channels=1, classes=10))
    #models.append(HQNN_Parallel(model_name='HQNN_Parallel_32', devices=hparams.devices, max_epochs=32, in_channels=1, classes=10))
    #models.append(HQNN_Quanv(model_name='HQNN_Quanv_8', devices=hparams.devices, max_epochs=8, in_channels=1, classes=10))
    #models.append(PennyClassifier(model_name='PennyClassifier_8', devices=hparams.devices, max_epochs=8, in_dim=28, in_channels=1, classes=10))

    # TODO For HQNN_Quanv, try a bigger kernel (7-8 qubits?) to test training speed
    # TODO for pennyclassifier, need to implement a PCA or some kind of dimensionality reduction to reduce the input size

    for model in models:
        model.fit(train_dataset, val_dataset)
        plot_training_result(model, len(train_dataset.dataset), len(val_dataset.dataset))
    # TODO Not working at the moment
    #plot_models_together(models)

    # TODO visualising the filters and feature maps!
    # TODO how to do that with a VQC model??

    # Performing attacks
    test_dataset = data_manager.load_dataset(train=False)
    epsilons = [x/100 for x in range(0, 105, 5)]
    fgsm = FGSMAttack(test_dataset)
    pgd = PGDAttack(test_dataset)
    
    accuracies = {}
    for source_model in models:
        evaluation_results = {}
        accuracies[source_model.model_name] = {model.model_name: [] for model in models}

        og_preds = source_model(test_dataset.data.unsqueeze(1).to(torch.float))
        og_acc = calculate_accuracy(og_preds, test_dataset.targets)
        print("Accuracy of model : ", og_acc)
        for epsilon in epsilons:
            adv_examples = fgsm.attack(source_model, epsilon)
            evaluation_results[epsilon] = adv_examples
        
            for target_model in models:
                adv_pred = target_model(adv_examples)
                adv_acc = calculate_accuracy(adv_pred, test_dataset.targets[:len(adv_examples)])
                accuracies[source_model.model_name][target_model.model_name].append(adv_acc)
                if source_model.model_name == target_model.model_name:
                     print(f"Accuracy of model {source_model.model_name} with epsilon {epsilon} : {adv_acc}")
        visualize_attacks_results(source_model, evaluation_results)
        plot_attack_effect_on_accuracy(f"attacks_{source_model.model_name}", models, epsilons, accuracies[source_model.model_name])
    plot_attack_effect_on_accuracy("attacks_ALL", models, epsilons, {model.model_name: accuracies[model.model_name][model.model_name] for model in models})

# Implement the parser (Make my own parser)
if __name__ == "__main__":
    logs_manager = LoggingService()
    # TODO add helpers for the ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-d", "--devices", type=int, default=1, help='TODO')
    parser.add_argument("-w", "--workers", type=int, default=0, help='TODO')
    args = parser.parse_args()
    start = time.time()
    main(args)
    end = time.time()
    print(f"Execution time: {end - start}")
    logs_manager.save_file(f"Execution time: {end - start}", "stats.txt")

    # TODO so basically how I see things, we should have a function you call to setup the
    # models, in it you should pass :
    # - the dataset and dataset configuration (batch size, num_workers, etc..)
    # - the model class
    # - the model configuration
    # - the training/optimization configuration

    # TODO load experiments (experiments > exp_name.json)
    # TODO in experiments you have a list of model configurations each placed in 
    # TODO models > model_name > configuration.json

    # TODO add a smoke_test arg that run the experiment on a reduced dataset to check if everything is working

    # TODO add seeding for the experiments!!

    # TODO Should implement some sanity checks, so usefull when testing everything...

    # TODO Quantum MLPMixer

    # TODO add my resnet16, MLP and MLPMixer to my models

    # TODO Gradient flow visualisation

    # TODO Graph flow of model visualisation

    # TODO add checkpoints

    # TODO UNDER EACH EXPERIMENT FOLDER THERE IS A SUB FOLDER WITH A NUMBER SAYING A SEED OF THAT RUN

    # TODO What would really be interesting beyond just looking at curves if it resist or not
    # Would be to visualise how the point are placed in the hillbert space where are the frontiers
    # and how attacking a certain VQC is going to change the position of the data points in the hilbert space


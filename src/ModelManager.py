import logging
from datetime import datetime
import os
import torch
from pytorch_lightning import Trainer

from visualisation.AttacksVisualisation import plot_accuracy_vs_epsilon
from visualisation.TrainingVisualisation import plot_training_result, plot_models_together

from Configuration import TENSORBOARD_DIR

from pytorch_lightning.loggers import TensorBoardLogger

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelManager:
    def __init__(self, max_epochs=10):
        self.models = []
        self.attacks = []

        self.max_epochs = max_epochs
        
        # Create a directory to store the results
        timestring = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
        self.root_dir = f"results/{timestring}"
        os.makedirs(self.root_dir, exist_ok=True)


    def register_model(self, model_name, model, configuration):
        logger_rootdir = f"{self.root_dir}/{TENSORBOARD_DIR}"
        logger = TensorBoardLogger(logger_rootdir, name="logs_test_"+model_name)
        trainer = Trainer(logger=logger, max_epochs=self.max_epochs, default_root_dir=logger_rootdir)
        self.models.append({'name': model_name, 'model': model, 'configuration': configuration, 'trainer': trainer})
        logging.info(f"Model '{model_name}' registered.")
        return self.models

    def register_attack(self, attack_name, attack_function, epsilon):
        self.attacks.append({'name': attack_name, 'function': attack_function, 'epsilon': epsilon})
        logging.info(f"Attack '{attack_name}' registered with epsilon {epsilon}.")
        return self.attacks

    def batch_train(self, train_loader, val_loader):
        results = {}
        for model in self.models:
            model_name = model['name']
            logging.info(f"Training model '{model_name}'...")
            trainer = model['trainer']
            model_instance = model['model']
            trainer.fit(model_instance, train_loader, val_loader)
            results[model_name] = trainer.logged_metrics
        return results

    def batch_evaluate(self, test_loader):
        results = {}
        for model in self.models:
            model_name = model['name']
            logging.info(f"Evaluating model '{model_name}'...")
            trainer = model['trainer']
            model_instance = model['model']
            results[model_name] = trainer.test(model_instance, dataloaders=test_loader)[0]
        return results

    def batch_attack(self, test_loader):
        results = {}
        for model in self.models:
            model_name = model['name']
            results[model_name] = {'epsilons': [], 'accuracies': [], 'examples': []}
            for attack in self.attacks:
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

    def batch_save(self, training_results, evaluation_results, attack_results):
        """Visualize the training results and models together."""

        for model in self.models:
            model_name = model['name']
            model_instance = model['model']
            model_path = f"{self.root_dir}/{model_name}"
            os.makedirs(model_path)

            # Save the model
            model_file = f"{model_path}/model.pt"
            torch.save(model_instance.state_dict(), model_file)

            # Plot the training results
            plot_training_result(model_path, model, training_results[model_name])

            # Plot results for each attack
            if attack_results is not None:
                for attack_name, result in attack_results[model_name].items():
                    plot_accuracy_vs_epsilon(self.root_dir, model_name, attack_name, result['accuracies'], result['epsilons'])

        plot_models_together(self.root_dir, training_results)


class ExampleConfiguration:
    def __init__(self, optimiser, init_lr, loss_function, max_epochs=10, batch_size=64, gpus=None):
        self.optimiser = optimiser
        self.init_lr = init_lr
        self.loss_function = loss_function
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.gpus = gpus
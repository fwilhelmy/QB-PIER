import logging
from datetime import datetime
import os
import torch
from pytorch_lightning import Trainer

from visualisation.AttacksVisualisation import plot_accuracy_vs_epsilon
from visualisation.TrainingVisualisation import plot_training_result, plot_models_together

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

import foolbox as fb

from Configuration import TENSORBOARD_DIR

from pytorch_lightning.loggers import TensorBoardLogger

from art.estimators.classification import PyTorchClassifier
from attacks.AttackFactory import AttackFactory

from foolbox import PyTorchModel, accuracy, samples

import numpy as np


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelsManager:
    def __init__(self, max_epochs=10):
        self.models = []
        self.max_epochs = max_epochs
        
        # Create a directory to store the results
        timestring = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
        self.root_dir = f"results/{timestring}"
        os.makedirs(self.root_dir, exist_ok=True)

    def register_model(self, model_name, model, configuration):
        logger_rootdir = f"{self.root_dir}/{TENSORBOARD_DIR}"
        logger = TensorBoardLogger(logger_rootdir, name="logs_test_"+model_name)
        trainer = Trainer(logger=logger, max_epochs=self.max_epochs, default_root_dir=logger_rootdir)

        # Create an ART classifier for this model
        classifier = PyTorchClassifier(
            model=model,
            loss=configuration.loss_function,
            optimizer=configuration.optimiser,
            input_shape=(1, 28, 28),  # Example for MNIST; adjust for your dataset
            nb_classes=10,  # Adjust based on your number of classes
        )

        # Store the model along with its configuration, trainer, and classifier
        self.models.append({'name': model_name, 'model': model, 'configuration': configuration, 'trainer': trainer, 'classifier': classifier, 'attacks': []})
        logging.info(f"Model '{model_name}' registered.")
        return self.models

    def register_attack(self, attack_name, epsilon, **kwargs):
        for model in self.models:
            classifier = model['classifier']

            # Use the attack factory to create the attack
            attack = AttackFactory(attack_name, classifier, epsilon, **kwargs)
            # TODO Link the batch size and device to the global configs

            # Register the attack for this model
            model['attacks'].append({'name': f'{attack_name}', 'function': attack, 'epsilon': epsilon})
            logging.info(f"Attack '{attack_name}' registered for model '{model['name']}' with epsilon {epsilon}.")

        # TODO The attacks should be a list in the ModelManager and we make the attacks instances in the batch_attack method
        # Such as that we make the classifier and the attack instance in the batch attack method
        # Reducing space and time complexity
        # return model['attacks']

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

    def batch_attack(self, dataset):
        results = {}
        for model in self.models:
            model_name = model['name']
            model['model'].eval()
            logging.info(f"Attacking model '{model_name}'...")
            results[model_name] = {'epsilons': [], 'accuracies': [], 'examples': []}
            for attack in model['attacks']:  # Access attacks specific to the model
                attack_name = attack['name']
                attack_function = attack['function']
                epsilon = attack['epsilon']
                logging.info(f"Performing '{attack_name}' on model '{model_name}' with epsilon {epsilon}...")
                
                # TODO Maybe cleanup the logging into something cleaner
                # TODO Maybe add a progress bar for the attacks
                # TODO Maybe change instead of unsqueezing maybe should change the config of the classifier
                # adv_examples = attack_function.generate(x=dataset.data.unsqueeze(1).numpy())
                fmodel = fb.PyTorchModel(model['model'], bounds=(0, 255))
                images = dataset.data.unsqueeze(1).float()
                labels = dataset.targets

                attack = fb.attacks.FGSM()

                x_test_adv, clipped_adv, success = attack(fmodel, images, labels, epsilons=[epsilon])
                #attack = FastGradientMethod(estimator=model['classifier'], eps=epsilon)
                #x_test_adv = attack.generate(x=dataset.data.unsqueeze(1).numpy())

                # Step 7: Evaluate the ART classifier on adversarial test examples
                # y_test = model['classifier'].predict(dataset.data.unsqueeze(1).numpy())
                # predictions = model['classifier'].predict(x_test_adv)
                # accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
                # print(f"Accuracy on adversarial test examples for epsilon {epsilon}: {(accuracy * 100)}%")

                robust_accuracy = 1 - success.float32().mean(axis=-1)
                print("robust accuracy for perturbations with")
                for eps, acc in zip([epsilon], robust_accuracy):
                    print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

                # we can also manually check this
                # we will use the clipped advs instead of the raw advs, otherwise
                # we would need to check if the perturbation sizes are actually
                # within the specified epsilon bound
                print()
                print("we can also manually check this:")
                print()
                print("robust accuracy for perturbations with")
                for eps, advs_ in zip([epsilon], clipped_adv):
                    acc2 = accuracy(fmodel, advs_, labels)
                    print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
                    print("    perturbation sizes:")
                    perturbation_sizes = (advs_ - images).norms.linf(axis=(1, 2, 3)).numpy()
                    print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
                    if acc2 == 0:
                        break

                acc, examples = self.evaluate_attack(model['classifier'], dataset, x_test_adv)
                results[model_name]['epsilons'].append(epsilon)
                results[model_name]['accuracies'].append(acc)
                results[model_name]['examples'].append(examples)
        return results

    def evaluate_attack(self, classifier, dataset, adv_examples):
        examples = {'original_images': [], 'perturbed_images': [], 'original_preds': [], 'perturbed_preds': []}
        original_images = dataset.data.unsqueeze(1).cpu().numpy()
        perturbed_images = adv_examples
        labels = dataset.targets.cpu().numpy()
            
        original_preds = classifier.predict(original_images)
        perturbed_preds = classifier.predict(perturbed_images)

        total = dataset.data.size(0)
        correct = (perturbed_preds.max(1) == labels).sum()
        accuracy = correct / total

        examples['original_images'] = original_images
        examples['perturbed_images'] = perturbed_images
        examples['original_preds'] = original_preds.argmax(axis=1)
        examples['perturbed_preds'] = perturbed_preds.argmax(axis=1)

        return accuracy, examples

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
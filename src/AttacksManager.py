import logging
import foolbox as fb

class AttacksManager:
    def __init__(self):
        self.attacks = []

    def register_attack(self, attack_name, epsilon, **kwargs):
        """
        Registers a new attack.

        Args:
            attack_name (str): The name of the attack.
            epsilon (float): The epsilon value for the attack.
            **kwargs: Additional parameters for the attack.
        """
        # Append the attack details to the attacks list
        self.attacks.append({
            'name': attack_name,
            'epsilon': epsilon,
            'args': kwargs  # Store additional arguments if any
        })
        logging.info(f"Attack '{attack_name}' registered with epsilon {epsilon}.")

    def attack_model(self, model, dataset):
        model_results = []
        fmodel = fb.PyTorchModel(model['classifier'], bounds=(0, 1))
        images, labels = dataset.data, dataset.targets

        for attack in self.attacks:
            attack_name = attack['name']
            epsilon = attack['epsilon']
            kwargs = attack['args']

            if attack_name == 'PGD':
                attack_instance = fb.attacks.ProjectedGradientDescent(**kwargs)
            else:
                logging.warning(f"Attack '{attack_name}' is not supported. Skipping.")
                continue

            # Generate adversarial examples
            x_test_adv, clipped_adv, success = attack_instance(fmodel, images, labels, epsilons=[epsilon])

            # Calculate robust accuracy
            robust_accuracy = 1 - success.float32().mean().item()
            logging.info(f"Robust accuracy for '{attack_name}' with epsilon {epsilon}: {robust_accuracy * 100:.2f}%")

            # Evaluate attack effectiveness
            evaluation_results = self.evaluate_attack(model['classifier'], dataset, x_test_adv)
            model_results.append({
                'attack_name': attack_name,
                'epsilon': epsilon,
                'robust_accuracy': robust_accuracy,
                'evaluation_results': evaluation_results
            })
        
        return model_results

    def evaluate_attack(self, classifier, dataset, adv_examples):
        results = {
            'perturbed_images': adv_examples,
            'original_preds': [],
            'perturbed_preds': [],
            'original_accuracy': 0.0,
            'perturbed_accuracy': 0.0,
            'accuracy_difference': 0.0
        }
        labels = dataset.targets.cpu().numpy()
        
        # Predict original and adversarial examples
        original_preds = classifier.predict(dataset.data.unsqueeze(1).numpy())
        perturbed_preds = classifier.predict(adv_examples)

        results['original_preds'] = original_preds
        results['perturbed_preds'] = perturbed_preds

        # Calculate accuracy
        original_accuracy = (original_preds == labels).mean()
        perturbed_accuracy = (perturbed_preds == labels).mean()
        accuracy_difference = original_accuracy - perturbed_accuracy

        results['original_accuracy'] = original_accuracy
        results['perturbed_accuracy'] = perturbed_accuracy
        results['accuracy_difference'] = accuracy_difference

        return results

    def batch_attack(self, models, dataset):
        all_results = {}
        for model in models:
            model_name = model['name']
            logging.info(f"Running attacks for model '{model_name}'")
            results = self.attack_model(model, dataset)
            all_results[model_name] = results
        return all_results
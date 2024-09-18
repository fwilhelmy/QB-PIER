import os
import torch
from datetime import datetime
from visualisation.AttacksVisualisation import plot_accuracy_vs_epsilon
from visualisation.TrainingVisualisation import plot_training_result, plot_models_together

class LogsManager:
    def __init__(self):
        # Create a directory to store the results
        timestring = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
        self.root_dir = f"results/{timestring}"
        os.makedirs(self.root_dir, exist_ok=True)

    def get_rootdir(self):
        """Return the root directory of the logs."""
        return self.root_dir

    def save_model(self, model_name, model_instance):
        """Save the model to the specified path."""
        model_path = f"{self.root_dir}/{model_name}"
        os.makedirs(model_path, exist_ok=True)
        model_file = f"{model_path}/model.pt"
        torch.save(model_instance.state_dict(), model_file)
        print(f"Model '{model_name}' saved at '{model_file}'.")

    def generate_graphics(self, model_name, training_results, evaluation_results=None, attack_results=None):
        """Generate and save all graphics for training, test, and attack results."""
        model_path = f"{self.root_dir}/{model_name}"
        
        self._generate_training_graphics(model_path, model_name, training_results)
        if evaluation_results is not None:
            self._generate_test_results_graphics(model_path, model_name, evaluation_results)
        if attack_results is not None:
            self._generate_attack_results_graphics(model_path, model_name, attack_results)

        self._generate_custom_graphics()
        plot_models_together(self.root_dir, {model_name: training_results})

    def _generate_training_graphics(self, model_path, model_name, training_results):
        """Generate graphics for training results."""
        plot_training_result(model_path, model_name, training_results)

    def _generate_test_results_graphics(self, model_path, model_name, test_results):
        """Generate graphics for test results."""
        # Implement test results visualization if needed
        pass

    def _generate_attack_results_graphics(self, model_path, model_name, attack_results):
        """Generate graphics for attack results."""
        if attack_results is not None:
            for attack_name, result in attack_results.items():
                plot_accuracy_vs_epsilon(self.root_dir, model_name, attack_name, result['accuracies'], result['epsilons'])

    def _generate_custom_graphics(self):
        """Placeholder for generating custom graphics."""
        pass

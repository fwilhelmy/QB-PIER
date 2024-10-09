import os
import torch
from datetime import datetime
from visualisation.AttacksVisualisation import plot_accuracy_vs_epsilon
from visualisation.TrainingVisualisation import plot_training_result, plot_models_together

class LoggingService:
    _instance = None    

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):  # Ensure the initialization code runs only once
            self.initialized = True
            
            # Create a directory to store the results
            timestring = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
            self.root_dir = f"results/{timestring}"
            os.makedirs(self.root_dir, exist_ok=True)

    def get_rootdir(self):
        """Return the root directory of the logs."""
        return self.root_dir
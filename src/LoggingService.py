import os
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger

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
            self._root_dir = f"results/{timestring}"
            os.makedirs(self._root_dir, exist_ok=True)

    def register_model(self, model_name: str):
        model_dir = f"{self._root_dir}/{model_name}"
        os.makedirs(model_dir, exist_ok=True)

    def get_root_dir(self):
        """Return the root directory of the logs."""
        return self._root_dir

    def create_tb_logger(self, model_name: str):
        """Create a TensorBoard logger for the given model."""
        return TensorBoardLogger(f"{self._root_dir}/tensorboard", name=model_name)
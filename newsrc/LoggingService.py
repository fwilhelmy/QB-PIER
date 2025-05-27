import os
from datetime import datetime

class LoggingService:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, root_dir=None):
        if not hasattr(self, 'initialized'):  # Ensure the initialization code runs only once
            self.initialized = True
            
            # Create a directory to store the results
            timestring = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
            root_dir = root_dir or "results"
            self._root_dir = f"{root_dir}/{timestring}"
            os.makedirs(self._root_dir, exist_ok=True)
            print(f"Tensorboard path: {self._root_dir}/tensorboard")

    def get_root_dir(self):
        """Return the root directory of the logs."""
        return self._root_dir
    
    def save_figure(self, figure, title, model=None):
        figure_dir = f"{self._root_dir}/" + (model.model_name if model else "")
        figure.savefig(f"{figure_dir}/{title}.png")

    def save_file(self, file_content, file_name, model=None):
        file_dir = f"{self._root_dir}/" + (model.model_name if model else "")
        with open(f"{file_dir}/{file_name}", "w") as file:
            file.write(file_content)


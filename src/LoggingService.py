import os
from datetime import datetime
import matplotlib.pyplot as plt
from Configuration import LOGS_DIR
from pytorch_lightning.utilities import rank_zero_only

class LoggingService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, exp_dir=None):
        if not hasattr(self, 'initialized'):  # Ensure the initialization code runs only once
            self.initialized = True

            # Create a directory to store the results
            timestring = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
            self._root_dir = os.path.join(LOGS_DIR, exp_dir, timestring) if exp_dir else os.path.join(LOGS_DIR, timestring)
            self._init_root_dir()
        elif exp_dir is not None:
            print(f"[Warning] Cannot change the experiment directory after initialization. Current root directory: {self._root_dir}")

    @rank_zero_only
    def _init_root_dir(self):
        os.makedirs(self._root_dir, exist_ok=True)
        
    def get_root_dir(self):
        """Return the root directory of the logs."""
        return self._root_dir
    
    @rank_zero_only
    def save_figure(self, figure, title, model=None):
        figure_dir = model.log_dir if model else f"{self._root_dir}/"
        os.makedirs(figure_dir, exist_ok=True)
        figure.savefig(f"{figure_dir}/{title}.png")
        plt.close(figure)

    @rank_zero_only
    def save_file(self, file_content, file_name, model=None):
        file_dir = model.log_dir if model else f"{self._root_dir}/"
        os.makedirs(file_dir, exist_ok=True)
        with open(f"{file_dir}/{file_name}", "w") as file:
            file.write(file_content)
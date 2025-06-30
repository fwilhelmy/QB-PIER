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
    def save_figure(self, figure, title, log_dir=None):
        fig_dir = os.path.join(self._root_dir, log_dir) if log_dir else self._root_dir
        os.makedirs(fig_dir, exist_ok=True)
        figure.savefig(os.path.join(fig_dir, f"{title}.png"))
        plt.close(figure)

    @rank_zero_only
    def save_file(self, content, title, log_dir=None):
        file_dir = os.path.join(self._root_dir, log_dir) if log_dir else self._root_dir
        os.makedirs(file_dir, exist_ok=True)
        with open(os.path.join(file_dir, title), "w") as file:
            file.write(content)
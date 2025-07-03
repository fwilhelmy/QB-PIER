import os
import json
import atexit
import inspect
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from Configuration import LOGS_DIR
from pytorch_lightning.utilities import rank_zero_only
from . import __version__

class LoggingService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, exp_dir=None):
        if not hasattr(self, 'initialized'):
            self.initialized = True

            timestring = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
            self._root_dir = os.path.join(LOGS_DIR, exp_dir, timestring) if exp_dir else os.path.join(LOGS_DIR, timestring)
            self._start_time = datetime.now()
            self._info = {
                "pipeline_version": __version__,
                "start_time": self._start_time.isoformat(),
                "cuda_available": torch.cuda.is_available(),
            }
            self._capture_configuration()
            self._init_root_dir()
            atexit.register(self.finalize)
        elif exp_dir is not None:
            print(f"[Warning] Cannot change the experiment directory after initialization. Current root directory: {self._root_dir}")

    @rank_zero_only
    def _init_root_dir(self):
        os.makedirs(self._root_dir, exist_ok=True)

    def _capture_configuration(self):
        try:
            import Configuration as cfg
        except Exception:
            return

        cfg_dict = {}
        for name in dir(cfg):
            if name.startswith("_"):
                continue
            value = getattr(cfg, name)
            if inspect.isfunction(value) or inspect.ismodule(value):
                continue
            try:
                json.dumps(value)
                cfg_dict[name] = value
            except TypeError:
                cfg_dict[name] = str(value)

        self._info["configuration"] = cfg_dict
        
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

    def finalize(self):
        if getattr(self, "_finalized", False):
            return
        self._finalized = True
        end_time = datetime.now()
        self._info["end_time"] = end_time.isoformat()
        self._info["duration_seconds"] = (end_time - self._start_time).total_seconds()

        # Attempt to store current git commit
        try:
            import subprocess
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).decode().strip()
            self._info["git_commit"] = commit
        except Exception:
            pass

        self.save_file(json.dumps(self._info, indent=2), "experiment_info.json")

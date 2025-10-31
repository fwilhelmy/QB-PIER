import logging
import warnings
import os
import json
from datetime import datetime
from typing import Union
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import torch
from pytorch_lightning.utilities import rank_zero_only

log = logging.getLogger(__name__)
from __init__ import __version__

class LoggingService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, root_dir: str = "logs", *args, **kwargs):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._root_dir = root_dir
            self._init_root_dir()
            self._start_time = datetime.now()
            self._info = {
                "pipeline_version": __version__,
                "start_time": self._start_time.isoformat(),
                "cuda_available": torch.cuda.is_available(),
            }
            logging.captureWarnings(True)
            warnings.simplefilter("default")

    @property
    def root_dir(self) -> str:
        return self._root_dir

    @rank_zero_only
    def _init_root_dir(self):
        os.makedirs(self._root_dir, exist_ok=True)

    @rank_zero_only
    def save_figure(self, figure, title, path=None):
        fig_dir = os.path.join(self._root_dir, path) if path else self._root_dir
        os.makedirs(fig_dir, exist_ok=True)
        figure.savefig(os.path.join(fig_dir, f"{title}.png"))
        plt.close(figure)

    @rank_zero_only
    def save_file(self, content, title, path=None):
        file_dir = os.path.join(self._root_dir, path) if path else self._root_dir
        os.makedirs(file_dir, exist_ok=True)
        with open(os.path.join(file_dir, title), "w", encoding="utf-8") as file:
            file.write(content)

    # @contextmanager
    # def timeit(self, name: str, step: Optional[int] = None):
    #     start = time.time()
    #     yield
    #     elapsed = time.time() - start
    #     self.log_scalar("time", elapsed, group=name)
    #     if step is not None:
    #         self.log_scalar("step", step, group=name)

    @rank_zero_only
    def save_experiment(self):
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

        self.save_file(json.dumps(self._info, indent=2), "training.info")

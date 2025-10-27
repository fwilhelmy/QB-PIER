import logging
import os
import json
from datetime import datetime
from typing import Union

import matplotlib.pyplot as plt
import torch
from hydra.utils import log
from pytorch_lightning.utilities import rank_zero_only

from . import __version__


class _ErrorAccumulator(logging.Handler):
    """Collect error-level log records emitted during an experiment."""

    def __init__(self, collector):
        super().__init__(level=logging.ERROR)
        self._collector = collector
        # Keep formatting simple so stack traces are preserved via Formatter.format.
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - logging internals
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - defensive fallback
            message = record.getMessage()
        self._collector(message)

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
            self._errors = []
            self._attach_error_handler()

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

    def _attach_error_handler(self) -> None:
        """Attach an in-memory handler that mirrors error logs to ``errors.log``."""
        root_logger = logging.getLogger()
        if not any(isinstance(handler, _ErrorAccumulator) for handler in root_logger.handlers):
            self._error_handler = _ErrorAccumulator(self._errors.append)
            root_logger.addHandler(self._error_handler)
        else:
            # Reuse the existing handler to continue accumulating within the singleton.
            for handler in root_logger.handlers:
                if isinstance(handler, _ErrorAccumulator):
                    self._error_handler = handler
                    break

    @rank_zero_only
    def log_error(self, error: Union[Exception, str]):
        message = str(error)
        log.error(message)

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

        if self._errors:
            self.save_file("\n".join(self._errors), "errors.log")

        self.save_file(json.dumps(self._info, indent=2), "training.info")

import json
import logging
import os
import warnings
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import torch
from pytorch_lightning.utilities import rank_zero_only

from DiscordWebhook import DiscordWebhook

log = logging.getLogger(__name__)
from __init__ import __version__


@contextmanager
@rank_zero_only
def ExperimentManager(log_path: Optional[str] = None):
    start_time = datetime.now()
    info = {
        "pipeline_version": __version__,
        "start_time": start_time.isoformat(),
        "cuda_available": torch.cuda.is_available(),
    }

    yield info

    end_time = datetime.now()
    info["end_time"] = end_time.isoformat()
    info["duration_seconds"] = (end_time - start_time).total_seconds()

    try:
        import subprocess

        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT)
            .decode().strip()
        )
        info["git_commit"] = commit
    except Exception:
        log.error("Could not retrieve git commit hash.")

    output_dir = log_path or "."
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "experiment.info")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(json.dumps(info, indent=2))

# Singleton to configure and capture logs in a central directory.
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
            logging.captureWarnings(True)
            warnings.simplefilter("default")
            self._init_discord_logging()

    @property
    def root_dir(self) -> str:
        return self._root_dir

    @rank_zero_only
    def _init_root_dir(self):
        os.makedirs(self._root_dir, exist_ok=True)

    @rank_zero_only
    def _init_discord_logging(self) -> None:
        if getattr(self, "_discord_handler", None):
            return

        handler = DiscordWebhook()
        handler.setLevel(logging.WARNING)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(handler)
        self._discord_handler = handler

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from LoggingService import LoggingService
from metrics import ExperimentProfiler, MetricsContainer
from models.ModelFactory import ModelFactory


class BaseRepository(ABC):
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def profile(self, *args, **kwargs):
        ...

    @abstractmethod
    def save(self, *args, **kwargs):
        ...

    @abstractmethod
    def load(self, *args, **kwargs):
        ...
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import logging
import pandas as pd

@dataclass
class AttackResult:
    seed: int
    source_model: str
    target_model: str
    epsilon: float
    accuracy: float
    num_samples: int
    predictions_path: str | None
    gradients_path: str | None

    def to_record(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "source_model": self.source_model,
            "target_model": self.target_model,
            "epsilon": self.epsilon,
            "accuracy": self.accuracy,
            "num_samples": self.num_samples,
            "predictions_path": self.predictions_path,
            "gradients_path": self.gradients_path,
        }
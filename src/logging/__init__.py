"""
logging — custom training-time instrumentation.

Contains:
  - MetricsContainer: collects and aggregates metrics per step/epoch
  - EpochTimingCallback: measures epoch durations
  - TrainingLogger: orchestrates TensorBoard/CSV logging
"""

__version__ = "0.1.0"

# expose these classes at package level
from src.logging.MetricsContainer import MetricsContainer
from src.logging.EpochTimingCallback import EpochTimingCallback
from src.logging.TrainingLogger import TrainingLogger
from src.logging.LoggingService import LoggingService

# what gets imported with `from logging import *`
__all__ = [
    "MetricsContainer",
    "EpochTimingCallback",
    "TrainingLogger",
    "LoggingService",
]

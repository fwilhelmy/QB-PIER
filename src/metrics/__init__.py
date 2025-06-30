"""
logging — custom training-time instrumentation.

Contains:
  - MetricsContainer: collects and aggregates metrics per step/epoch
  - EpochTimingCallback: measures epoch durations
  - TrainingLogger: orchestrates TensorBoard/CSV logging
"""

__version__ = "0.1.0"

# expose these classes at package level
from .MetricsContainer import MetricsContainer
from .EpochTimingCallback import EpochTimingCallback
from .TrainingLogger import TrainingLogger
from .LoggingService import LoggingService

# what gets imported with `from logging import *`
__all__ = [
    "MetricsContainer",
    "EpochTimingCallback",
    "TrainingLogger",
    "LoggingService",
]

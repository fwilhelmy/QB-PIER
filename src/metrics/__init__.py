"""
logging — custom training-time instrumentation.

Contains:
  - MetricsContainer: collects and aggregates metrics per step/epoch
  - TrainingLogger: orchestrates TensorBoard/CSV logging
"""

# expose these classes at package level
from .MetricsContainer import MetricsContainer
from .TrainingLogger import TrainingLogger
from .CompactDeviceStats import CompactDeviceStats

# what gets imported with `from logging import *`
__all__ = [
    "MetricsContainer",
    "TrainingLogger",
    "CompactDeviceStats",
]

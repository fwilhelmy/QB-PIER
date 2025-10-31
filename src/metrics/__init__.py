"""
logging — custom training-time instrumentation.

Contains:
  - MetricsContainer: collects and aggregates metrics per step/epoch
"""

__version__ = "0.2.0"

# expose these classes at package level
from .MetricsContainer import MetricsContainer
from .LoggingService import LoggingService

# what gets imported with `from logging import *`
__all__ = [
    "MetricsContainer",
    "LoggingService",
]

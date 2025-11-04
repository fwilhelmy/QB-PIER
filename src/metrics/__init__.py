"""
logging — custom training-time instrumentation.

Contains:
  - MetricsContainer: collects and aggregates metrics per step/epoch
"""

# expose these classes at package level
from .MetricsContainer import MetricsContainer
from .CompactDeviceStats import CompactDeviceStats
from .GradientsProfiler import GradientProfiler

# what gets imported with `from logging import *`
__all__ = [
    "MetricsContainer",
    "CompactDeviceStats",
    "GradientProfiler",
]

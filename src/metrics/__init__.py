"""
logging — custom training-time instrumentation.

Contains:
  - MetricsContainer: collects and aggregates metrics per step/epoch
"""

# what gets imported with `from logging import *`
__all__ = [
    "MetricsContainer",
    "CompactDeviceStats",
    "GradientProfiler",
    "ExperimentProfiler",
]


def __getattr__(name):
    if name == "MetricsContainer":
        from .MetricsContainer import MetricsContainer as cls

        return cls
    if name == "CompactDeviceStats":
        from .CompactDeviceStats import CompactDeviceStats as cls

        return cls
    if name == "GradientProfiler":
        from .GradientsProfiler import GradientProfiler as cls

        return cls
    if name == "ExperimentProfiler":
        from .ExperimentProfiler import ExperimentProfiler as cls

        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

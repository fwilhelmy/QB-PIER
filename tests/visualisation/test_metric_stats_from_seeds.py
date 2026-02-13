from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sys
import types

if "seaborn" not in sys.modules:
    sys.modules["seaborn"] = types.ModuleType("seaborn")

if "matplotlib.colors" not in sys.modules:
    colors_module = types.ModuleType("matplotlib.colors")
    colors_module.to_rgb = lambda value: value
    sys.modules["matplotlib.colors"] = colors_module

if "matplotlib.ticker" not in sys.modules:
    ticker_module = types.ModuleType("matplotlib.ticker")

    class _MaxNLocator:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    ticker_module.MaxNLocator = _MaxNLocator
    ticker_module.FuncFormatter = lambda func: func
    sys.modules["matplotlib.ticker"] = ticker_module


import pytest

from src.visualisation.utils import metric_stats_from_seeds


@dataclass
class _Event:
    wall_time: float
    value: float


@dataclass
class _SeedModel:
    metrics: dict[str, list[float]]
    log_dir: str


def _seed_dict() -> dict[str, _SeedModel]:
    return {
        "0": _SeedModel(
            metrics={"val/acc": [0.5, 0.6, 0.7], "val/step": [10, 20, 30]},
            log_dir="/tmp/run_0",
        ),
        "1": _SeedModel(
            metrics={"val/acc": [0.6, 0.7, 0.8], "val/step": [10, 20, 30]},
            log_dir="/tmp/run_1",
        ),
    }


def test_metric_stats_step_axis() -> None:
    x, mean, std = metric_stats_from_seeds(_seed_dict(), metric="acc", group="val", x_axis="step")

    np.testing.assert_array_equal(x, np.array([10, 20, 30]))
    np.testing.assert_allclose(mean, np.array([0.55, 0.65, 0.75]))
    np.testing.assert_allclose(std, np.array([0.05, 0.05, 0.05]))


@pytest.mark.parametrize("axis", ["epoch", "indice"])
def test_metric_stats_epoch_and_indice_axes(axis: str) -> None:
    x, mean, std = metric_stats_from_seeds(_seed_dict(), metric="acc", group="val", x_axis=axis)

    np.testing.assert_array_equal(x, np.array([0, 1, 2]))
    np.testing.assert_allclose(mean, np.array([0.55, 0.65, 0.75]))
    np.testing.assert_allclose(std, np.array([0.05, 0.05, 0.05]))


def test_metric_stats_time_axis_uses_tensorboard_events(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [_Event(wall_time=100.0, value=0.0), _Event(wall_time=120.0, value=0.0), _Event(wall_time=155.0, value=0.0)]

    monkeypatch.setattr("src.visualisation.utils.find_tensorboard_dir", lambda run_dir: Path(run_dir) / "tensorboard")
    monkeypatch.setattr("src.visualisation.utils.load_scalar_events", lambda tb_dir, scalar_tag: events)

    x, mean, std = metric_stats_from_seeds(_seed_dict(), metric="acc", group="val", x_axis="time")

    np.testing.assert_array_equal(x, np.array([0.0, 20.0, 55.0]))
    np.testing.assert_allclose(mean, np.array([0.55, 0.65, 0.75]))
    np.testing.assert_allclose(std, np.array([0.05, 0.05, 0.05]))


def test_metric_stats_invalid_axis() -> None:
    with pytest.raises(ValueError, match="Invalid x_axis"):
        metric_stats_from_seeds(_seed_dict(), metric="acc", group="val", x_axis="banana")

"""Basic smoke test for :mod:`layers.classical.MLP`."""

from __future__ import annotations

import torch

from layers.classical import MLP


def test_mlp_runs_forward_pass() -> None:
    layer = MLP(in_features=4, output_size=2, hidden_sizes=())
    x = torch.randn(3, 4)
    out = layer(x)
    assert out.shape == (3, 2)

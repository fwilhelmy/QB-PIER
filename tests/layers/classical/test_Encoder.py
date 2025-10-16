"""Basic smoke test for :mod:`layers.classical.Encoder`."""

from __future__ import annotations

import torch

from layers.classical import Encoder


def test_encoder_projects_to_last_hidden_size() -> None:
    layer = Encoder(input_size=(1, 4, 4), hidden_sizes=(6, 5))
    x = torch.randn(2, 1, 4, 4)
    out = layer(x)
    assert out.shape == (2, 5)

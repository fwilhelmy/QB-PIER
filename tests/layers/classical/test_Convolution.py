"""Basic smoke test for :mod:`layers.classical.Convolution`."""

from __future__ import annotations

import torch

from layers.classical import Convolution


def test_convolution_produces_flat_output() -> None:
    layer = Convolution(input_size=(1, 8, 8), conv_channels=(2, 4))
    x = torch.randn(2, 1, 8, 8)
    out = layer(x)
    assert out.shape == (2, layer.output_size)

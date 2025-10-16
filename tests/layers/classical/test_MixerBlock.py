"""Basic smoke test for :mod:`layers.classical.MixerBlock`."""

from __future__ import annotations

import torch

from layers.classical import MixerBlock


def test_mixer_block_preserves_shape() -> None:
    block = MixerBlock(dim=8, seq_len=4)
    x = torch.randn(2, 4, 8)
    out = block(x)
    assert out.shape == x.shape

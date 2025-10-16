"""Basic smoke test for :mod:`layers.classical.ResBlock`."""

from __future__ import annotations

import torch

from layers.classical import ResBlock


def test_resblock_changes_channel_count() -> None:
    block = ResBlock(in_planes=3, planes=6, stride=2)
    x = torch.randn(1, 3, 16, 16)
    out = block(x)
    assert out.shape == (1, 6, 8, 8)

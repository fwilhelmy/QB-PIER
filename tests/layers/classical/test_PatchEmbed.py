"""Basic smoke test for :mod:`layers.classical.PatchEmbed`."""

from __future__ import annotations

import torch

from layers.classical import PatchEmbed


def test_patch_embed_creates_expected_patch_count() -> None:
    layer = PatchEmbed(img_size=8, patch_size=4, in_chans=3, embed_dim=16)
    x = torch.randn(2, 3, 8, 8)
    out = layer(x)
    assert out.shape == (2, 4, 16)

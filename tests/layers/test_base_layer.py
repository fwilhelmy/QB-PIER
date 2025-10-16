"""Tests for the shared :mod:`layers.BaseLayer` utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from layers import BaseLayer


class _ToyLayer(BaseLayer):
    """Minimal concrete layer used to exercise ``BaseLayer`` helpers."""

    def __init__(self) -> None:
        super().__init__(in_features=4, output_size=2)
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.linear(x)


def test_freeze_and_unfreeze_toggle_gradients() -> None:
    layer = _ToyLayer()
    layer.freeze()
    assert all(not p.requires_grad for p in layer.parameters())

    layer.unfreeze()
    assert all(p.requires_grad for p in layer.parameters())


def test_load_roundtrip(tmp_path: Path) -> None:
    layer = _ToyLayer()
    checkpoint = tmp_path / "toy.pt"

    # Persist the pristine parameters and then perturb them.
    torch.save(layer.state_dict(), checkpoint)
    with torch.no_grad():
        layer.linear.weight.add_(1.0)

    layer.load(str(checkpoint))
    assert torch.allclose(layer.linear.weight, torch.load(checkpoint)["linear.weight"])

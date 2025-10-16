"""Basic smoke test for :mod:`layers.quantum.ansatz.BasicEntanglingAnsatz`."""

from __future__ import annotations

import pytest

pytest.importorskip("pennylane")

from layers.quantum.ansatz import BasicEntanglingAnsatz


def test_basic_entangling_ansatz_reports_weight_shapes() -> None:
    ansatz = BasicEntanglingAnsatz(input_size=2, n_layers=1)
    shapes = ansatz.weight_shapes
    assert "weights" in shapes

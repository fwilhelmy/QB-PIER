"""Basic smoke test for :mod:`layers.quantum.ansatz.StrongEntanglingAnsatz`."""

from __future__ import annotations

import pytest

pytest.importorskip("pennylane")

from layers.quantum.ansatz import StrongEntanglingAnsatz


def test_strong_entangling_ansatz_reports_weight_shapes() -> None:
    ansatz = StrongEntanglingAnsatz(input_size=3, n_layers=1)
    assert "weights" in ansatz.weight_shapes

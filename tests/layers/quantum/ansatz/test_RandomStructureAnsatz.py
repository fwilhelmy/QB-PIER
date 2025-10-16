"""Basic smoke test for :mod:`layers.quantum.ansatz.RandomStructureAnsatz`."""

from __future__ import annotations

import pytest

pytest.importorskip("pennylane")

from layers.quantum.ansatz import RandomStructureAnsatz


def test_random_structure_ansatz_has_weights_entry() -> None:
    ansatz = RandomStructureAnsatz(input_size=2, n_layers=1)
    assert "weights" in ansatz.weight_shapes

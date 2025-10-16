"""Basic smoke test for :mod:`layers.quantum.ansatz.NoEntanglingAnsatz`."""

from __future__ import annotations

import pytest

pytest.importorskip("pennylane")

from layers.quantum.ansatz import NoEntanglingAnsatz


def test_no_entangling_ansatz_weight_shape_matches_layers() -> None:
    ansatz = NoEntanglingAnsatz(input_size=2, n_layers=3)
    assert ansatz.weight_shapes["weights"] == (3, ansatz.n_qubits, 3)

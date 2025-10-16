"""Basic smoke test for :mod:`layers.quantum.ansatz.ReUpAnsatzWrapper`."""

from __future__ import annotations

import pytest

pytest.importorskip("pennylane")

from layers.quantum.ansatz import ReUpAnsatzWrapper


def test_reup_ansatz_wrapper_provides_weights() -> None:
    ansatz = ReUpAnsatzWrapper(input_size=2, n_layers=2)
    assert ansatz.weight_shapes == {"weights": (2, ansatz.n_qubits)}

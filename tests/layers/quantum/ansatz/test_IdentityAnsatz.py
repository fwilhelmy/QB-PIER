"""Basic smoke test for :mod:`layers.quantum.ansatz.IdentityAnsatz`."""

from __future__ import annotations

import pytest

pytest.importorskip("pennylane")

from layers.quantum.ansatz import IdentityAnsatz


def test_identity_ansatz_has_no_parameters() -> None:
    ansatz = IdentityAnsatz(input_size=1, n_layers=1)
    assert ansatz.weight_shapes == {"weights": (0,)}

"""Basic smoke test for :mod:`layers.quantum.ansatz.ResidualAnsatzWrapper`."""

from __future__ import annotations

import pytest

from layers.quantum.ansatz import BasicEntanglingAnsatz, ResidualAnsatzWrapper


def test_residual_wrapper_is_not_ready_for_basic_ansatz() -> None:
    with pytest.raises(TypeError):
        ResidualAnsatzWrapper(BasicEntanglingAnsatz)

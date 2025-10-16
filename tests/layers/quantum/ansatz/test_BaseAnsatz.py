"""Basic smoke test for :mod:`layers.quantum.ansatz.BaseAnsatz`."""

from __future__ import annotations

import pytest

pytest.importorskip("pennylane")

from layers.quantum.ansatz import BaseAnsatz


class _SimpleAnsatz(BaseAnsatz):
    @property
    def weight_shapes(self) -> dict:
        return {"weights": (self.n_layers, self.n_qubits)}

    def ansatz(self, weights):  # pragma: no cover - no circuit for smoke test
        return weights


def test_base_ansatz_tracks_qubits() -> None:
    ansatz = _SimpleAnsatz(input_size=3, n_layers=2, embedding="none")
    assert ansatz.n_qubits == 3
    assert ansatz.weight_shapes == {"weights": (2, 3)}

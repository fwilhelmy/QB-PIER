"""Basic smoke test for :mod:`layers.quantum.ansatz.AnsatzFactory`."""

from __future__ import annotations

import pytest

pytest.importorskip("pennylane")

from layers.quantum.ansatz import AnsatzFactory, BasicEntanglingAnsatz


def test_ansatz_factory_creates_registered_class() -> None:
    ansatz = AnsatzFactory.create(
        "BasicEntanglingAnsatz",
        input_size=2,
        n_layers=1,
    )
    assert isinstance(ansatz, BasicEntanglingAnsatz)

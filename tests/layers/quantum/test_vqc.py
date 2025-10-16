"""Tests for the :class:`~layers.quantum.VQC` module."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from layers.quantum import VQC


def test_vqc_forward_returns_expected_shape() -> None:
    layer = VQC(input_size=2, n_layers=1)
    x = torch.randn(4, 2)
    out = layer(x)
    assert out.shape == (4, layer.output_size)


def test_vqc_respects_ansatz_configuration() -> None:
    layer = VQC(
        input_size=3,
        n_layers=2,
        ansatz={"class": "NoEntanglingAnsatz"},
    )
    assert layer.ansatz.__class__.__name__ == "NoEntanglingAnsatz"
    assert layer.output_size == layer.ansatz.n_qubits

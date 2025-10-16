"""Tests for :class:`layers.quantum.ParallelVQC`."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from layers.quantum import ParallelVQC


def test_parallel_vqc_concatenates_block_outputs() -> None:
    layer = ParallelVQC(
        input_size=6,
        n_qblocks=3,
        n_layers=1,
        ansatz={"class": "NoEntanglingAnsatz"},
    )
    x = torch.randn(2, 6)
    out = layer(x)
    assert out.shape == (2, layer.output_size)


def test_parallel_vqc_caps_number_of_blocks() -> None:
    layer = ParallelVQC(input_size=4, n_qblocks=10, n_layers=1)
    assert layer.n_blocks == 4  # Cannot create more blocks than features
    x = torch.randn(1, 4)
    out = layer(x)
    assert out.shape[1] == layer.output_size

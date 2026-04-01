import numpy as np
import pytest

from src.layers.quantum.ansatz import BasicEntanglingAnsatz


def test_circuit_depth_returns_int():
    ansatz = BasicEntanglingAnsatz(input_size=2, n_layers=1)
    depth = ansatz.circuit_depth()
    assert isinstance(depth, int)
    assert depth > 0


def test_circuit_num_gates_returns_int():
    ansatz = BasicEntanglingAnsatz(input_size=2, n_layers=1)
    num_gates = ansatz.circuit_num_gates()
    assert isinstance(num_gates, int)
    assert num_gates > 0

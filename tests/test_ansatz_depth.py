import numpy as np
import pytest

from models.quantum.ansatz import BasicEntanglingAnsatz


def test_circuit_depth_returns_int():
    ansatz = BasicEntanglingAnsatz(input_size=2, n_layers=1)
    depth = ansatz.circuit_depth()
    assert isinstance(depth, int)
    assert depth > 0

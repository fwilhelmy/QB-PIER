import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from src.models.quantum.Quanv2d import Quanv2d


def test_quanv2d_forward_shape():
    layer = Quanv2d(in_channels=1, out_channels=4, kernel_size=2, stride=2)
    x = torch.randn(2, 1, 4, 4)
    out = layer(x)
    assert out.shape == (2, 4, 2, 2)


def test_quanv2d_wrong_channels():
    layer = Quanv2d(in_channels=1, out_channels=2, kernel_size=2)
    x = torch.randn(1, 3, 4, 4)
    with pytest.raises(ValueError):
        layer(x)

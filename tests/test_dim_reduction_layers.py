import pytest
import torch

sklearn = pytest.importorskip("sklearn")
from src.models.dimensionality_reduction import (
    PCALayer,
    KernelPCALayer,
    RandomProjectionLayer,
    TruncatedSVDLayer,
    UMAPLayer,
)


@pytest.mark.parametrize("layer_cls", [PCALayer, KernelPCALayer, RandomProjectionLayer, TruncatedSVDLayer])
def test_sklearn_layers_shape(layer_cls):
    x = torch.randn(10, 20)
    layer = layer_cls(n_components=5)
    layer.fit(x)
    out = layer(x)
    assert out.shape == (10, 5)


def test_umap_layer_shape():
    umap = pytest.importorskip("umap")
    x = torch.randn(15, 30)
    layer = UMAPLayer(n_components=3)
    layer.fit(x)
    out = layer(x)
    assert out.shape == (15, 3)


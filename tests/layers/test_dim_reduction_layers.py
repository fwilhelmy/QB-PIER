import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
sklearn = pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "notebooks" / "dimVQC" / "DimensionalityReductionLayer.py"
SPEC = importlib.util.spec_from_file_location("dim_layers", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
DimensionalityReductionLayer = MODULE.DimensionalityReductionLayer


@pytest.mark.parametrize("method", ["pca", "kernel_pca", "random_projection", "truncated_svd"])
def test_sklearn_layers_shape(method):
    x = torch.randn(10, 20)
    layer = DimensionalityReductionLayer(method=method, n_components=5)
    layer.fit(x)
    out = layer(x)
    assert out.shape == (10, 5)


def test_umap_layer_shape():
    umap = pytest.importorskip("umap")
    x = torch.randn(15, 30)
    layer = DimensionalityReductionLayer(method="umap", n_components=3)
    layer.fit(x)
    out = layer(x)
    assert out.shape == (15, 3)

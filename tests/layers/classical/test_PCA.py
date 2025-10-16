"""Basic smoke test for :mod:`layers.classical.PCA`."""

from __future__ import annotations

from importlib import import_module

import torch
from sklearn.decomposition import PCA as SklearnPCA


_pca_module = import_module("layers.classical.PCA")
_LayerPCA = _pca_module.PCA


def test_pca_fit_and_transform(monkeypatch) -> None:
    monkeypatch.setattr(_pca_module, "PCA", SklearnPCA)

    layer = _LayerPCA(input_size=(1, 4, 4), n_components=2)
    data = torch.randn(6, 1, 4, 4)
    layer.fit(data)
    out = layer(data[:2])
    assert out.shape == (2, 2)

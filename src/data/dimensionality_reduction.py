"""Torch layers for common dimensionality reduction techniques."""

from typing import Optional
import torch
from torch import nn


class _BaseReducer(nn.Module):
    """Base class for wrappers around scikit-learn/UMAP reducers."""

    def __init__(self, n_components: int, **kwargs) -> None:
        super().__init__()
        self.n_components = n_components
        self.kwargs = kwargs
        self.reducer = None

    def fit(self, x: torch.Tensor) -> "_BaseReducer":
        if self.reducer is None:
            raise RuntimeError("Reducer not initialized")
        data = x.detach().cpu().numpy()
        self.reducer.fit(data)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reducer is None:
            raise RuntimeError("Reducer not initialized")
        data = x.detach().cpu().numpy()
        transformed = self.reducer.transform(data)
        return torch.as_tensor(transformed, device=x.device, dtype=x.dtype)


class PCALayer(_BaseReducer):
    """Principal Component Analysis layer."""

    def __init__(self, n_components: int, **kwargs) -> None:
        super().__init__(n_components, **kwargs)
        from sklearn.decomposition import PCA
        self.reducer = PCA(n_components=n_components, **kwargs)


class KernelPCALayer(_BaseReducer):
    """Kernel PCA layer."""

    def __init__(self, n_components: int, kernel: str = "rbf", **kwargs) -> None:
        super().__init__(n_components, kernel=kernel, **kwargs)
        from sklearn.decomposition import KernelPCA
        self.reducer = KernelPCA(n_components=n_components, kernel=kernel, **kwargs)


class RandomProjectionLayer(_BaseReducer):
    """Gaussian random projection layer."""

    def __init__(self, n_components: int, **kwargs) -> None:
        super().__init__(n_components, **kwargs)
        from sklearn.random_projection import GaussianRandomProjection
        self.reducer = GaussianRandomProjection(n_components=n_components, **kwargs)


class TruncatedSVDLayer(_BaseReducer):
    """Truncated SVD layer."""

    def __init__(self, n_components: int, **kwargs) -> None:
        super().__init__(n_components, **kwargs)
        from sklearn.decomposition import TruncatedSVD
        self.reducer = TruncatedSVD(n_components=n_components, **kwargs)


class UMAPLayer(_BaseReducer):
    """UMAP dimensionality reduction layer."""

    def __init__(self, n_components: int, **kwargs) -> None:
        super().__init__(n_components, **kwargs)
        import umap
        self.reducer = umap.UMAP(n_components=n_components, **kwargs)

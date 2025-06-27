"""Torch layer for common dimensionality reduction techniques via a single class."""
from typing import Dict, Type
import torch
from torch import nn

from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
import umap

# Module-level registry of supported reducers
REGISTRY: Dict[str, Type] = {
    'pca': PCA,
    'kernel_pca': KernelPCA,
    'truncated_svd': TruncatedSVD,
    'random_projection': GaussianRandomProjection, # Johnson-Lindenstrauss lemma
    'umap': umap.UMAP,
}

class DimensionalityReductionLayer(nn.Module):
    """
    A PyTorch layer wrapper for various sklearn/UMAP reducers.

    Parameters
    ----------
    method : str
        Which reducer to use. One of REGISTRY.keys().
    n_components : int
        Number of target dimensions.
    **kwargs
        Passed through to the underlying reducer's constructor.
    """
    def __init__(self, method: str, n_components: int, **kwargs) -> None:
        super().__init__()
        key = method.lower()
        if key not in REGISTRY:
            raise ValueError(
                f"Unknown reduction method {method!r}. "
                f"Choose from {list(REGISTRY.keys())}"
            )
        ReducerClass = REGISTRY[key]
        self.reducer = ReducerClass(n_components=n_components, **kwargs)

        # store for repr/name
        self.method = key
        # e.g. "kernel_pca" → ["kernel","pca"] → ["Kernel","Pca"] → "KernelPcaLayer"
        parts = [w.capitalize() for w in key.split('_')]
        self._layer_name = ''.join(parts) + 'Layer'

        self.n_components = n_components

    def _get_name(self) -> str:
        # Overrides nn.Module _get_name() so repr(self) shows e.g. "PCALayer(...)"
        return self._layer_name

    def fit(self, x: torch.Tensor) -> "DimensionalityReductionLayer":
        """Fit reducer to data (CPU numpy array)."""
        data = x.detach().cpu().numpy()
        self.reducer.fit(data)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input tensor through the fitted reducer."""
        if not hasattr(self.reducer, 'transform'):
            raise RuntimeError(f"Reducer '{self.method}' not fitted or has no transform.")
        data = x.detach().cpu().numpy()
        transformed = self.reducer.transform(data)
        return torch.as_tensor(transformed, device=x.device, dtype=x.dtype)

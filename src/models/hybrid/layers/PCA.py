from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from sklearn.decomposition import PCA

from ..HybridLayer import HybridLayer
from ..HybridLayerFactory import HybridLayerFactory


@HybridLayerFactory.encoder("pca")
class PCA(HybridLayer):
    """Hybrid layer applying Principal Component Analysis."""

    def __init__(
        self,
        input_size: tuple[int, int, int],
        n_components: int,
    ) -> None:
        super().__init__(input_size=input_size)
        self.input_dim = int(np.prod(input_size))
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.output_size = n_components

    def fit(self, data: Tensor) -> None:
        flat = data.view(data.size(0), -1).detach().cpu().numpy()
        self.pca.fit(flat)

    def forward(self, x: Tensor) -> Tensor:
        flat = x.view(x.size(0), -1).detach().cpu().numpy()
        transformed = self.pca.transform(flat)
        return torch.tensor(transformed, device=x.device, dtype=x.dtype)

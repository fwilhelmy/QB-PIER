from __future__ import annotations

import torch
from torch.nn import Module


class HybridLayer(Module):
    """Base class for hybrid model components.

    Provides convenience methods to freeze or unfreeze parameters and to
    load state dictionaries from disk. Encoders and classifiers inherit from
    this class so that these utilities are available on all hybrid layers.
    """

    def freeze(self) -> None:
        """Disable gradients for all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Enable gradients for all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def load(self, path: str) -> None:
        """Load parameters from ``path`` into this layer."""
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state)


__all__ = ["HybridLayer"]

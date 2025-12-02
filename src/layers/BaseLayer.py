from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch.nn import Module

class BaseLayer(Module):
    """Common base class for hybrid encoders and classifiers.

    The original ``BaseEncoder`` and ``BaseClassifier`` classes shared a
    number of utility attributes such as ``input_size`` and ``output_size``.
    Refactoring them into a single ``BaseLayer``
    ensures that every concrete encoder and classifier exposes the same
    lightweight interface without duplicating bookkeeping logic.
    """

    def __init__(
        self,
        *,
        input_size: Optional[Union[int, Tuple[int, int, int]]] = None,
        output_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size

    # ------------------------------------------------------------------
    # Dimension bookkeeping
    # ------------------------------------------------------------------
    @property
    def input_size(self) -> Optional[Union[int, Tuple[int, int, int]]]:
        """Dimensions expected by the layer before any processing."""

        return self._input_size

    @input_size.setter
    def input_size(self, value: Union[int, Tuple[int, int, int]]) -> None:
        self._input_size = value

    @property
    def output_size(self) -> Optional[int]:
        """Number of features produced by the layer, if known."""

        return self._output_size

    @output_size.setter
    def output_size(self, value: int) -> None:
        self._output_size = value

    # ------------------------------------------------------------------
    # Convenience helpers shared by all hybrid layers
    # ------------------------------------------------------------------
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


__all__ = ["BaseLayer"]

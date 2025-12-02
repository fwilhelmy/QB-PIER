from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.nn import Module

class BaseLayer(Module):
    """Common base class for hybrid encoders and classifiers.

    The original ``BaseEncoder`` and ``BaseClassifier`` classes shared a
    number of utility attributes such as ``input_size``, ``in_features`` and
    ``output_size``.  Refactoring them into a single ``BaseLayer``
    ensures that every concrete encoder and classifier exposes the same
    lightweight interface without duplicating bookkeeping logic.
    """

    def __init__(
        self,
        *,
        input_size: Optional[Tuple[int, int, int]] = None,
        in_features: Optional[int] = None,
        output_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._input_size = input_size
        self._in_features = in_features
        self._output_size = output_size

    # ------------------------------------------------------------------
    # Dimension bookkeeping
    # ------------------------------------------------------------------
    @property
    def input_size(self) -> Optional[Tuple[int, int, int]]:
        """Spatial input dimensions expected by the layer, if applicable."""

        return self._input_size

    @input_size.setter
    def input_size(self, value: Tuple[int, int, int]) -> None:
        self._input_size = value

    @property
    def in_features(self) -> Optional[int]:
        """Number of features consumed by the layer, if defined."""

        return self._in_features

    @in_features.setter
    def in_features(self, value: int) -> None:
        self._in_features = value

    @property
    def output_size(self) -> Optional[int]:
        """Number of features produced by the layer, if known."""

        return self._output_size

    @output_size.setter
    def output_size(self, value: int) -> None:
        self._output_size = value

__all__ = ["BaseLayer"]

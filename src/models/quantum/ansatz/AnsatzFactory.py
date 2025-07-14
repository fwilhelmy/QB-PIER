from __future__ import annotations

from typing import Type

from .BaseAnsatz import BaseAnsatz
from .BasicEntanglingAnsatz import BasicEntanglingAnsatz


class AnsatzFactory:
    """Instantiate ansatz circuits from configuration parameters."""

    def __init__(self, ansatz_cls: Type[BaseAnsatz] = BasicEntanglingAnsatz, **params) -> None:
        self.ansatz_cls = ansatz_cls
        self.params = params

    def build(self, input_size: int, n_layers: int, embedding: str):
        """Return an initialized ansatz instance."""
        return self.ansatz_cls(input_size, n_layers, embedding, **self.params)

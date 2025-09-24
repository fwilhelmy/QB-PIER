# TODO Will need much more work and to manually define my ansatzs to be able to do something like this...
# This is more an epic than anything else, but I want to keep it here for now.

from __future__ import annotations
from . import AnsatzFactory, BaseAnsatz

@AnsatzFactory.ansatz("ReUpAnsatzWrapper")
class ReUpAnsatzWrapper(BaseAnsatz):
    """Wrapper for the ReUp Ansatz to be used in a VQC."""

    def __init__(
        self, 
        input_size: int, 
        n_layers: int, 
        embedding: str = "angle"
    ):
        super().__init__(input_size, n_layers, embedding)

    @property
    def weight_shapes(self) -> dict:
        """Define the shape of the weights for the ReUp Ansatz."""
        return {"weights": (self.n_layers, self.n_qubits)}

    def ansatz(self, weights):
        """Define the ReUp Ansatz."""
        # The actual implementation of the ReUp Ansatz would go here.
        # This is a placeholder for the actual quantum circuit definition.
        pass

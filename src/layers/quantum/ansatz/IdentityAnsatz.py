from . import AnsatzFactory, BaseAnsatz

@AnsatzFactory.ansatz("IdentityAnsatz")
class IdentityAnsatz(BaseAnsatz):
    def __init__(
        self,
        input_size: int,
        n_layers: int,
        embedding: str = "angle"
    ):
        super().__init__(input_size, n_layers, embedding)

    @property
    def weight_shapes(self) -> dict:
        """
        Define the shape of the weights for the Basic Entangling Ansatz.
        Returns a dictionary with weight names and their shapes.
        """
        return {"weights": (0,)} # No trainable parameters

    def ansatz(self, weights):
        pass # Intentionally no entangling layers

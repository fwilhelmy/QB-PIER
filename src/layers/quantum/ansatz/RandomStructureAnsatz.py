import pennylane as qml
from . import AnsatzFactory, BaseAnsatz

@AnsatzFactory.ansatz("RandomStructureAnsatz")
class RandomStructureAnsatz(BaseAnsatz):
    def __init__(
        self,
        input_size: int,
        n_layers: int,
        embedding: str = "angle",
        quantum_config=None,
        ratio_imprim: float = 0.75, # Ratio of rotation 1 qubit gates to cnots
        seed: int = 42,
    ):
        super().__init__(input_size, n_layers, embedding, quantum_config=quantum_config)
        self.ratio_imprim = ratio_imprim
        self.seed = seed

    @property
    def weight_shapes(self) -> dict:
        return {"weights": qml.RandomLayers.shape(n_layers=self.n_layers, n_rotations=3)}

    def ansatz(self, weights):
        qml.RandomLayers(
            weights,
            wires=range(self.n_qubits),
            ratio_imprim=self.ratio_imprim,
            seed=self.seed,
        )

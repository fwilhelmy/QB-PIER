from overrides import override
import torch, pennylane as qml
from . import AnsatzFactory, BaseAnsatz

def ResidualAnsatzWrapper(ansatz_cls: type[BaseAnsatz]) -> type[BaseAnsatz]:
    class ResidualAnsatz(ansatz_cls):
        def __init__(self, input_size, n_layers, **ansatz_kwargs):
            # Initialize the parent ansatz with no embedding
            ansatz_kwargs["embedding"] = "none"
            super().__init__(
                input_size=input_size,
                n_layers=n_layers,
                **ansatz_kwargs,
            )
    
        @property
        def weight_shapes(self) -> dict:
            # Call the parent property to get the base shapes
            weight_shapes = super().weight_shapes["weights"]
            return {"weights": weight_shapes, "bias": weight_shapes}

        @override
        def circuit(self, inputs, weights, bias):
            # Tile once per layer
            stacked = torch.hstack(
                [inputs] * self.n_layers
            )  # (batch_size, in_dim * in_dim * n_reup)
            tiled = stacked.reshape(
                self.weight_shapes["weights"]
            )  # (num_layers, n_qubits, n_rotations)

            # Apply the ansatz with modulated weights using the concrete implementation from ansatz_cls
            ansatz_cls.ansatz(self, weights * tiled + bias)

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        @override
        def build_qlayer(self):
            # same as before, but fix the bias key
            init = {
                "weights": 0.1 * torch.rand(self.weight_shapes["weights"]),
                "bias": 0.1 * torch.rand(self.weight_shapes["bias"]),
            }
            return self.create_qtorchlayer(
                self.n_qubits, self.circuit, self.weight_shapes, init
            )

    ResidualAnsatz.__name__ = f"Residual{ansatz_cls.__name__}"
    return ResidualAnsatz


AnsatzFactory.register("ResidualAnsatzWrapper", ResidualAnsatzWrapper)

from overrides import override
import torch, pennylane as qml
from utils import create_qtorchlayer

# TODO layer(template, depth, *args, **kwargs)

def ResidualAnsatzWrapper(ansatz_cls):
    class ResidualAnsatz(ansatz_cls):
        def __init__(self, input_size, n_layers, **ansatz_kwargs):
            # Initialize the parent ansatz with no embedding
            ansatz_kwargs["embedding"] = "none"
            ansatz_cls.__init__(
                self,
                input_size=input_size,
                n_layers=n_layers,
                **ansatz_kwargs,
            )

        @property
        def weight_shapes(self) -> dict:
            # Call the parent property to get the base shapes
            weight_shapes = ansatz_cls.weight_shapes.fget(self)["weights"]
            return {"weights": weight_shapes, "bias": weight_shapes}

        @override
        def circuit(self, inputs, weights, bias):
            # Tile once per layer
            stacked = torch.hstack([inputs] * self.n_layers) # (batch_size, in_dim * in_dim * n_reup)
            tiled  = stacked.reshape(self.weight_shapes["weights"]) # (num_layers, n_qubits, n_rotations)

            # Apply the ansatz with modulated weights
            ansatz_cls.ansatz(self, weights * tiled + bias)

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        @override
        def build_qlayer(self):
            # same as before, but fix the bias key
            init = {
                "weights": 0.1 * torch.rand(self.weight_shapes["weights"]),
                "bias":    0.1 * torch.rand(self.weight_shapes["bias"]),
            }
            return create_qtorchlayer(self.n_qubits, self.circuit, self.weight_shapes, init)

    ResidualAnsatz.__name__ = f"Residual{ansatz_cls.__name__}"
    return ResidualAnsatz

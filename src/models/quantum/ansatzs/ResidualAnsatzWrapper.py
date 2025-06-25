import torch, pennylane as qml
from src.models.quantum.ansatzs.BaseAnsatz import BaseAnsatz
from src.utils import create_qtorchlayer

def ResidualAnsatzWrapper(ansatz_cls):
    class ResidualAnsatz(ansatz_cls):
        def __init__(self, n_qubits, n_layers, rotation="Y", **ansatz_kwargs):
            # Initialize the parent ansatz with no embedding
            ansatz_cls.__init__(
                self,
                n_qubits=n_qubits,
                n_layers=n_layers,
                embedding="none",
                rotation=rotation,
                **ansatz_kwargs,
            )

            weight_shapes = self.weight_shapes["weights"]
            self.weight_shapes = {"weights": weight_shapes, "bias": weight_shapes}

        def circuit(self, inputs, weights):
            W = weights.pop("weights")
            B = weights.pop("bias")

            # Tile once per layer
            stacked = torch.hstack([inputs] * self.num_reup) # (batch_size, in_dim * in_dim * n_reup)
            tiled  = stacked.reshape(self.weight_shapes["weights"]) # (num_layers, n_qubits, n_rotations)

            # Apply the ansatz with modulated weights
            ansatz_cls.ansatz(self, W * tiled + B)

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        def build_qlayer(self):
            # same as before, but fix the bias key
            init = {
                "weights": 0.1 * torch.rand(self.weight_shapes["weights"]),
                "bias":    0.1 * torch.rand(self.weight_shapes["bias"]),
            }
            return create_qtorchlayer(self.n_qubits, self.circuit, self.weight_shapes, init)

    ResidualAnsatz.__name__ = f"Residual{ansatz_cls.__name__}"
    return ResidualAnsatz

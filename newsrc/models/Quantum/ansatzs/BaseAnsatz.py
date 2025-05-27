import pennylane as qml
import numpy as np
from utils import get_qnode

class BaseAnsatz:
    """
    Base class for all quantum Ansatz circuits.
    Handles embedding; subclasses must override `ansatz()` to define gate structure.
    """
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        embedding: str = "angle",
        rotation: str = "X",
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.embedding = embedding.lower()
        self.rotation = rotation
        self.weight_shapes = None

    def _embed(self, inputs):
        """
        Apply the input embedding onto the wires.
        Supports 'angle' and 'amplitude'.
        """
        if self.embedding == "angle":
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation=self.rotation)
        elif self.embedding == "amplitude":
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)
        else:
            raise ValueError(f"Unknown embedding method: {self.embedding}")

    def ansatz(self, weights):
        """
        Override in subclasses to implement the gate structure.
        """
        raise NotImplementedError("Subclasses must implement the ansatz() method.")

    def circuit(self, inputs, weights):
        """
        Full QNode circuit: embedding + ansatz + measurements.
        """
        self._embed(inputs)
        self.ansatz(weights)
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]
    
    def visualize(self, device):
        """
        Visualize the circuit diagram using matplotlib.
        No need to supply inputs or weights; zeros are used by default.

        Args:
            device: PennyLane device (e.g., qml.device(...)).

        Returns:
            A matplotlib.Figure containing the circuit diagram.
        """
        # Default zero inputs and weights
        default_inputs = np.zeros(self.n_qubits)
        weight_shape = self.weight_shapes.get("weights", ())
        default_weights = np.zeros(weight_shape)

        qnode = get_qnode(self.circuit, self.n_qubits)
        mpl_drawer = qml.draw_mpl(qnode)
        fig = mpl_drawer(default_inputs, default_weights)
        return fig
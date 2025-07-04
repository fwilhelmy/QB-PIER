import pennylane as qml
import numpy as np
from utils import get_qnode, create_qtorchlayer
from Configuration import CIRCUIT_STYLE

class BaseAnsatz:
    """
    Base class for all quantum Ansatz circuits.
    Handles embedding; subclasses must override `ansatz()` to define gate structure.
    """
    weight_shapes: dict
    # TODO pass to constructor the weight shapes
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        embedding: str = "angle", # 'angle' or 'amplitude' or 'none'
        rotation: str = "X",
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.embedding = embedding.lower()
        self.rotation = rotation

    def _embed(self, inputs):
        """
        Apply the input embedding onto the wires.
        Supports 'angle' and 'amplitude'.
        """
        if self.embedding == "angle":
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation=self.rotation)
        elif self.embedding == "amplitude":
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True, pad_with=0.0)
        elif self.embedding != "none":
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

    def build_qlayer(self):
        self.qlayer = create_qtorchlayer(self.n_qubits, self.circuit, self.weight_shapes)
        return self.qlayer
    
    def visualize(self, expanded: bool = False):
        """
        Visualize the circuit diagram using matplotlib.

        Args:
            expanded (bool): If True, show fully decomposed gates.
                             If False, show high-level layer boxes.

        Returns:
            fig: A matplotlib.Figure containing the circuit diagram.
        """
        assert self.weight_shapes is not None, "Weight shapes must be defined before visualization."

        # Default zero inputs and weights
        default_inputs = np.zeros(self.n_qubits)
        weight_shape = self.weight_shapes.get("weights", ())
        default_weights = np.zeros(weight_shape)

        qnode = get_qnode(self.circuit, self.n_qubits)
        strategy = "device" if expanded else "gradient"
        mpl_drawer = qml.draw_mpl(qnode, style=CIRCUIT_STYLE, level=strategy)

        # Draw and return figure
        fig, _ = mpl_drawer(default_inputs, default_weights)
        return fig
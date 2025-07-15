import pennylane as qml
import numpy as np
from qdevice import QDEVICE, INTERFACE, QDIFF, CIRCUIT_STYLE
from abc import ABC, abstractmethod
from models.quantum.embedding import EmbeddingFactory


class BaseAnsatz(ABC):
    """Base class for all quantum Ansatz circuits."""
    def __init__(
        self,
        input_size: int,
        n_layers: int,
        embedding: str = "angle",
    ):
        self.n_layers = n_layers
        self.embedding, self.n_qubits = EmbeddingFactory(embedding, input_size)

    @property
    @abstractmethod
    def weight_shapes(self) -> dict:
        """
        Define the shape of the weights for the Ansatz.
        """
        raise NotImplementedError(
            "Subclasses must implement the weight_shapes property."
        )
        
    @property
    def qnode(self):
        dev = qml.device(QDEVICE, wires=self.n_qubits)
        return qml.QNode(self.circuit, dev, interface=INTERFACE, diff_method=QDIFF)

    @property
    def qlayer(self):
        if not hasattr(self, '_qlayer'):
            init_vals = None

            self._qlayer = qml.qnn.TorchLayer(
                qnode=self.qnode,
                weight_shapes=self.weight_shapes,
                init_method=init_vals
            )
        
        return self._qlayer

    @abstractmethod
    def ansatz(self, weights):
        """
        Define the Ansatz circuit.
        """
        raise NotImplementedError("Subclasses must implement the ansatz() method.")

    def circuit(self, inputs, weights):
        """
        Full QNode circuit: embedding + ansatz + measurements.
        """
        if self.embedding:
            self.embedding(inputs)
        self.ansatz(weights)
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

    def visualize(self, expanded: bool = False):
        """
        Visualize the circuit diagram using matplotlib.

        Args:
            expanded (bool): If True, show fully decomposed gates.
                             If False, show high-level layer boxes.

        Returns:
            fig: A matplotlib.Figure containing the circuit diagram.
        """
        assert (
            self.weight_shapes is not None
        ), "Weight shapes must be defined before visualization."

        # Default zero inputs and weights
        default_inputs = np.zeros(self.n_qubits)
        weight_shape = self.weight_shapes.get("weights", ())
        default_weights = np.zeros(weight_shape)

        strategy = "device" if expanded else "gradient"
        mpl_drawer = qml.draw_mpl(self.qnode, style=CIRCUIT_STYLE, level=strategy)

        # Draw and return figure
        fig, _ = mpl_drawer(default_inputs, default_weights)
        return fig
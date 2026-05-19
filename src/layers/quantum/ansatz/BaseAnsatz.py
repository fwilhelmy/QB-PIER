from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np
import pennylane as qml

from qdevice import QuantumDeviceConfig, normalize_quantum_config
from ..embedding import EmbeddingFactory


class BaseAnsatz(ABC):
    """Base class for all quantum Ansatz circuits."""

    def __init__(
        self,
        input_size: int,
        n_layers: int,
        embedding: str = "angle",
        quantum_config: Mapping[str, Any] | QuantumDeviceConfig | None = None,
    ):
        self.n_layers = n_layers
        self.quantum_config = normalize_quantum_config(quantum_config)

        if embedding.lower() == "none":
            self.embedding, self.n_qubits = None, input_size
        else:
            self.embedding = EmbeddingFactory.create(embedding, input_size=input_size)
            self.n_qubits = self.embedding.n_qubits

    @property
    @abstractmethod
    def weight_shapes(self) -> dict:
        """Define the shape of the weights for the Ansatz."""
        raise NotImplementedError(
            "Subclasses must implement the weight_shapes property."
        )

    @property
    def qnode(self):
        dev = qml.device(
            self.quantum_config.device,
            wires=self.n_qubits,
            shots=self.quantum_config.shots,
        )
        return qml.QNode(
            self.circuit,
            dev,
            interface=self.quantum_config.interface,
            diff_method=self.quantum_config.diff_method,
        )

    @property
    def qlayer(self):
        if not hasattr(self, "_qlayer"):
            init_vals = None
            self._qlayer = qml.qnn.TorchLayer(
                qnode=self.qnode,
                weight_shapes=self.weight_shapes,
                init_method=init_vals,
            )
        return self._qlayer

    @abstractmethod
    def ansatz(self, weights):
        """Define the Ansatz circuit."""
        raise NotImplementedError("Subclasses must implement the ansatz() method.")

    def circuit(self, inputs, weights):
        """Full QNode circuit: embedding + ansatz + optional noise + measurements."""
        if self.embedding:
            self.embedding(inputs)
        self.ansatz(weights)
        self._apply_noise()
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

    def _apply_noise(self) -> None:
        """Apply a small configurable noise block before measurement.

        This keeps noise explicit and local to quantum circuits. The supported
        options are deliberately simple and stable: depolarizing,
        phase_damping, amplitude_damping, bit_flip and phase_flip.
        """
        noise = self.quantum_config.noise
        if not noise.enabled:
            return

        name = str(noise.name).lower()
        probability = float(noise.probability)
        wires = range(self.n_qubits)

        for wire in wires:
            if name in {"depolarizing", "depolarizing_channel"}:
                qml.DepolarizingChannel(probability, wires=wire)
            elif name in {"phase_damping", "phasedamping"}:
                qml.PhaseDamping(probability, wires=wire)
            elif name in {"amplitude_damping", "amplitudedamping"}:
                qml.AmplitudeDamping(probability, wires=wire)
            elif name in {"bit_flip", "bitflip"}:
                qml.BitFlip(probability, wires=wire)
            elif name in {"phase_flip", "phaseflip"}:
                qml.PhaseFlip(probability, wires=wire)
            else:
                raise ValueError(
                    f"Unsupported quantum noise model '{noise.name}'. "
                    "Supported: depolarizing, phase_damping, amplitude_damping, bit_flip, phase_flip."
                )

    def visualize(self, expanded: bool = False):
        """Visualize the circuit diagram using matplotlib."""
        if self.weight_shapes is None:
            raise ValueError("Weight shapes must be defined before visualization.")

        default_inputs = np.zeros(self.n_qubits)
        weight_shape = self.weight_shapes.get("weights", ())
        default_weights = np.zeros(weight_shape)

        strategy = "device" if expanded else "gradient"
        mpl_drawer = qml.draw_mpl(
            self.qnode,
            style=self.quantum_config.circuit_style,
            level=strategy,
        )
        fig, _ = mpl_drawer(default_inputs, default_weights)
        return fig

    def circuit_depth(self) -> int:
        """Return the depth of the quantum circuit."""
        assert self.weight_shapes is not None, "Weight shapes must be defined before measuring depth."

        default_inputs = np.zeros(self.n_qubits)
        weight_shape = self.weight_shapes.get("weights", ())
        default_weights = np.zeros(weight_shape)

        specs = qml.specs(self.qnode)(default_inputs, default_weights)
        depth = specs.get("depth")
        if depth is None and "resources" in specs:
            depth = getattr(specs["resources"], "depth", 0)
        return int(depth or 0)

    def circuit_num_gates(self) -> int:
        """Return the total number of gates in the quantum circuit."""
        assert self.weight_shapes is not None, "Weight shapes must be defined before measuring gate count."

        default_inputs = np.zeros(self.n_qubits)
        weight_shape = self.weight_shapes.get("weights", ())
        default_weights = np.zeros(weight_shape)

        specs = qml.specs(self.qnode)(default_inputs, default_weights)
        num_gates = specs.get("num_operations")
        if num_gates is None and "resources" in specs:
            num_gates = getattr(specs["resources"], "num_gates", 0)
        return int(num_gates or 0)

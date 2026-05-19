from __future__ import annotations

"""Explicit PennyLane quantum-device and noise configuration.

This module intentionally does not load Hydra files or mutate global quantum
state. Training/attack entrypoints may still use Hydra, but they should pass the
resolved qdevice block down to the model, quantum layer, VQC and ansatz.

Noise is represented with PennyLane's native insertion-based API:
``qml.NoiseModel`` + ``qml.noise.add_noise``.  The project-level config remains
small and YAML-friendly, but the object used by the QNode is a real PennyLane
noise model instead of custom manual channel insertion inside the circuit.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

try:  # OmegaConf is present during Hydra runs, but keep this module lightweight.
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - optional dependency guard
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore


_NOISE_CHANNEL_NAMES = {
    "AmplitudeDamping",
    "BitFlip",
    "DepolarizingChannel",
    "PhaseDamping",
    "PhaseFlip",
}


@dataclass(frozen=True)
class NoiseConfig:
    """YAML-friendly wrapper for a PennyLane insertion-based noise model.

    The actual noise object used at runtime is built by :func:`build_noise_model`
    and is a PennyLane ``qml.NoiseModel``.

    Parameters
    ----------
    name:
        Noise channel name. Supported aliases include ``depolarizing``,
        ``phase_damping``, ``amplitude_damping``, ``bit_flip`` and
        ``phase_flip``.
    probability:
        Channel probability/gamma parameter.
    apply_to:
        Coarse condition for insertion. Supported values:
        ``all_ops`` / ``all_gates`` / ``all_wires``, ``single_qubit_gates``,
        ``two_qubit_gates``, or ``named_ops``.
    operations:
        Optional operation names/classes used when ``apply_to: named_ops``.
        Example: ["RX", "RY", "RZ"].
    wires:
        Optional wire restriction. If provided, the condition is combined with
        ``qml.noise.wires_in(wires)``.
    metadata:
        Optional metadata passed through to ``qml.NoiseModel``.
    """

    name: str | None = None
    probability: float = 0.0
    apply_to: str = "all_ops"
    operations: tuple[str, ...] = ()
    wires: tuple[int, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.name) and float(self.probability) > 0.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None | "NoiseConfig") -> "NoiseConfig":
        if isinstance(cfg, cls):
            return cfg
        if cfg is None:
            return cls()

        data = _to_plain_dict(cfg)
        name = data.get("name", data.get("model", data.get("channel", None)))
        enabled = data.get("enabled", None)
        probability = float(data.get("probability", data.get("p", data.get("gamma", 0.0))) or 0.0)

        # If enabled is explicitly false, preserve the name but disable by zeroing
        # the probability. This keeps config round-tripping simple.
        if enabled is False:
            probability = 0.0

        operations = data.get("operations", data.get("ops", ())) or ()
        if isinstance(operations, str):
            operations = (operations,)
        else:
            operations = tuple(str(op) for op in operations)

        wires = data.get("wires", None)
        if wires is not None:
            if isinstance(wires, int):
                wires = (wires,)
            elif isinstance(wires, str):
                wires = tuple(int(w.strip()) for w in wires.split(",") if w.strip())
            else:
                wires = tuple(int(w) for w in wires)

        metadata = data.get("metadata", {}) or {}
        return cls(
            name=name,
            probability=probability,
            apply_to=str(data.get("apply_to", "all_ops")),
            operations=operations,
            wires=wires,
            metadata=_to_plain_dict(metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "probability": self.probability,
            "apply_to": self.apply_to,
            "operations": list(self.operations),
            "wires": list(self.wires) if self.wires is not None else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class QuantumDeviceConfig:
    """Configuration needed to build PennyLane devices, QNodes and noise models."""

    device: str = "default.qubit"
    interface: str = "torch"
    diff_method: str = "backprop"
    circuit_style: str = "pennylane"
    shots: int | None = None
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None | "QuantumDeviceConfig") -> "QuantumDeviceConfig":
        if isinstance(cfg, cls):
            return cfg
        if cfg is None:
            return cls()

        data = _to_plain_dict(cfg)
        noise_data = data.get("noise", {}) or {}

        # Backward-compatible keys are intentionally supported so old config
        # files and old checkpoint hyperparameters remain loadable.
        shots = data.get("shots", data.get("SHOTS", None))
        if isinstance(shots, str) and shots.lower() in {"none", "null", "infinite", "inf"}:
            shots = None
        elif shots is not None:
            shots = int(shots)

        return cls(
            device=data.get("device", data.get("QDEVICE", "default.qubit")),
            interface=data.get("interface", data.get("INTERFACE", "torch")),
            diff_method=data.get("diff_method", data.get("QDIFF", "backprop")),
            circuit_style=data.get("circuit_style", data.get("CIRCUIT_STYLE", "pennylane")),
            shots=shots,
            noise=NoiseConfig.from_config(noise_data),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "interface": self.interface,
            "diff_method": self.diff_method,
            "circuit_style": self.circuit_style,
            "shots": self.shots,
            "noise": self.noise.as_dict(),
        }


def _to_plain_dict(cfg: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(cfg, QuantumDeviceConfig):
        return cfg.as_dict()
    if isinstance(cfg, NoiseConfig):
        return cfg.as_dict()
    if OmegaConf is not None and DictConfig is not None and isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)


def normalize_quantum_config(cfg: Mapping[str, Any] | None | QuantumDeviceConfig) -> QuantumDeviceConfig:
    """Return a concrete :class:`QuantumDeviceConfig` from dict/Hydra/dataclass input."""

    return QuantumDeviceConfig.from_config(cfg)


def create_device(config: Mapping[str, Any] | QuantumDeviceConfig | None, n_qubits: int):
    """Create a PennyLane device from an explicit config."""

    import pennylane as qml

    quantum_config = normalize_quantum_config(config)
    return qml.device(
        quantum_config.device,
        wires=n_qubits,
        shots=quantum_config.shots,
    )


def make_qnode(
    circuit: Callable[..., Any],
    config: Mapping[str, Any] | QuantumDeviceConfig | None,
    n_qubits: int,
):
    """Create a PennyLane QNode and attach native PennyLane noise if configured."""

    import pennylane as qml

    quantum_config = normalize_quantum_config(config)
    dev = create_device(quantum_config, n_qubits=n_qubits)
    qnode = qml.QNode(
        circuit,
        dev,
        interface=quantum_config.interface,
        diff_method=quantum_config.diff_method,
    )
    return add_configured_noise(qnode, quantum_config, n_qubits=n_qubits)


def add_configured_noise(qnode: Any, config: Mapping[str, Any] | QuantumDeviceConfig | None, n_qubits: int | None = None):
    """Attach a configured PennyLane ``NoiseModel`` to a QNode.

    Returns the original QNode when noise is disabled. This keeps callers simple
    and avoids making non-noisy simulation pay any extra transform cost.
    """

    import pennylane as qml

    quantum_config = normalize_quantum_config(config)
    noise_model = build_noise_model(quantum_config.noise, n_qubits=n_qubits)
    if noise_model is None:
        return qnode
    return qml.noise.add_noise(qnode, noise_model)


def build_noise_model(noise: Mapping[str, Any] | NoiseConfig | None, n_qubits: int | None = None):
    """Build a native PennyLane ``qml.NoiseModel`` from project config.

    This is intentionally the only place that knows how project YAML maps to
    PennyLane noise conditionals and noisy quantum functions.
    """

    import pennylane as qml

    noise_config = NoiseConfig.from_config(noise)
    if not noise_config.enabled:
        return None

    condition = _build_noise_condition(qml, noise_config, n_qubits=n_qubits)
    noise_fn = _build_noise_callable(qml, noise_config)
    return qml.NoiseModel({condition: noise_fn}, **dict(noise_config.metadata))


def _build_noise_callable(qml: Any, noise: NoiseConfig):
    channel = _resolve_noise_channel(qml, noise.name)
    probability = float(noise.probability)
    return qml.noise.partial_wires(channel, probability)


def _resolve_noise_channel(qml: Any, name: str | None):
    if not name:
        raise ValueError("Noise is enabled but no noise channel/model name was provided.")

    key = str(name).lower().replace("-", "_")
    channel_map = {
        "depolarizing": qml.DepolarizingChannel,
        "depolarizing_channel": qml.DepolarizingChannel,
        "phase_damping": qml.PhaseDamping,
        "phasedamping": qml.PhaseDamping,
        "amplitude_damping": qml.AmplitudeDamping,
        "amplitudedamping": qml.AmplitudeDamping,
        "bit_flip": qml.BitFlip,
        "bitflip": qml.BitFlip,
        "phase_flip": qml.PhaseFlip,
        "phaseflip": qml.PhaseFlip,
    }
    if key not in channel_map:
        supported = ", ".join(sorted(channel_map))
        raise ValueError(f"Unsupported quantum noise model '{name}'. Supported aliases: {supported}.")
    return channel_map[key]


def _build_noise_condition(qml: Any, noise: NoiseConfig, n_qubits: int | None = None):
    apply_to = noise.apply_to.lower().replace("-", "_")

    if apply_to in {"all", "all_ops", "all_gates", "all_wires"}:
        condition = _all_non_noise_ops_condition(qml)
    elif apply_to in {"single", "single_qubit", "single_qubit_gates"}:
        condition = _num_wires_condition(qml, 1)
    elif apply_to in {"two", "two_qubit", "two_qubit_gates"}:
        condition = _num_wires_condition(qml, 2)
    elif apply_to in {"named", "named_ops", "operations", "ops"}:
        condition = _named_ops_condition(qml, noise.operations)
    else:
        raise ValueError(
            f"Unsupported noise apply_to='{noise.apply_to}'. Supported: all_ops, "
            "single_qubit_gates, two_qubit_gates, named_ops."
        )

    if noise.wires is not None:
        condition = condition & qml.noise.wires_in(list(noise.wires))
    elif apply_to == "all_wires" and n_qubits is not None:
        condition = condition & qml.noise.wires_in(list(range(n_qubits)))

    return condition


def _all_non_noise_ops_condition(qml: Any):
    @qml.BooleanFn
    def all_non_noise_ops(op, **metadata):
        return _has_wires(op) and op.__class__.__name__ not in _NOISE_CHANNEL_NAMES

    return all_non_noise_ops


def _num_wires_condition(qml: Any, num_wires: int):
    @qml.BooleanFn
    def has_num_wires(op, **metadata):
        return (
            _has_wires(op)
            and len(op.wires) == num_wires
            and op.__class__.__name__ not in _NOISE_CHANNEL_NAMES
        )

    return has_num_wires


def _named_ops_condition(qml: Any, operations: Sequence[str]):
    if not operations:
        raise ValueError("noise.apply_to='named_ops' requires a non-empty noise.operations list.")

    op_classes = []
    for op_name in operations:
        try:
            op_classes.append(getattr(qml, str(op_name)))
        except AttributeError as exc:
            raise ValueError(f"Unknown PennyLane operation in noise.operations: '{op_name}'.") from exc

    condition = qml.noise.op_eq(op_classes[0])
    for op_cls in op_classes[1:]:
        condition = condition | qml.noise.op_eq(op_cls)
    return condition


def _has_wires(op: Any) -> bool:
    try:
        return len(op.wires) > 0
    except Exception:
        return False

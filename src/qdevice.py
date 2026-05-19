from __future__ import annotations

"""Explicit quantum-device configuration.

This module intentionally does not load Hydra files or mutate global quantum
state. Training/attack entrypoints may still use Hydra, but they should pass the
resolved qdevice block down to the model, quantum layer, VQC and ansatz.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping

try:  # OmegaConf is present during Hydra runs, but keep this module lightweight.
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - optional dependency guard
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore


@dataclass(frozen=True)
class NoiseConfig:
    """Simple in-circuit noise configuration for PennyLane channels."""

    name: str | None = None
    probability: float = 0.0
    apply_to: str = "all_wires"

    @property
    def enabled(self) -> bool:
        return bool(self.name) and float(self.probability) > 0.0


@dataclass(frozen=True)
class QuantumDeviceConfig:
    """Configuration needed to build PennyLane QNodes explicitly."""

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
        noise_data = _to_plain_dict(data.get("noise", {}) or {})

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
            noise=NoiseConfig(
                name=noise_data.get("name"),
                probability=float(noise_data.get("probability", 0.0) or 0.0),
                apply_to=noise_data.get("apply_to", "all_wires"),
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "interface": self.interface,
            "diff_method": self.diff_method,
            "circuit_style": self.circuit_style,
            "shots": self.shots,
            "noise": {
                "name": self.noise.name,
                "probability": self.noise.probability,
                "apply_to": self.noise.apply_to,
            },
        }


def _to_plain_dict(cfg: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(cfg, QuantumDeviceConfig):
        return cfg.as_dict()
    if OmegaConf is not None and DictConfig is not None and isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)


def normalize_quantum_config(cfg: Mapping[str, Any] | None | QuantumDeviceConfig) -> QuantumDeviceConfig:
    """Return a concrete :class:`QuantumDeviceConfig` from dict/Hydra/dataclass input."""

    return QuantumDeviceConfig.from_config(cfg)

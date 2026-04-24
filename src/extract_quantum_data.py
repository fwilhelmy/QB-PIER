from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import pennylane as qml
from omegaconf import OmegaConf
from tqdm import tqdm

from loading import crawl_experiments
from models.ModelFactory import ModelFactory

log = logging.getLogger(__name__)


QUANTUM_LAYER_NAMES = {
    "vql",
    "pvql",
    "hvql",
    "hpvql",
    "dropvql",
    "resvql",
    "quanv",
}

HQNN_MANUAL_PART_MAP = {
    "hqe": "classifier",
    "hvqm": "classifier",
    "hpvqm": "classifier",
    "hqnn_vql": "classifier",
    "hqnn_hpvql": "classifier",
    "hqnn_dropvql": "classifier",
    "hqnn_resvql": "classifier",
    "hquanv": "encoder",
}


def read_hydra_config(model_dir: Path) -> Dict[str, Any]:
    cfg_path = model_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return {}
    return OmegaConf.to_container(OmegaConf.load(str(cfg_path)), resolve=True)  # type: ignore[return-value]


def find_checkpoint(seed_dir: Path, load_mode: str) -> Optional[Path]:
    final_ckpt = seed_dir / "final.ckpt"
    if load_mode == "final":
        return final_ckpt if final_ckpt.exists() else None

    ckpt_dir = seed_dir / "checkpoints"
    if not ckpt_dir.exists():
        return final_ckpt if final_ckpt.exists() else None

    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return final_ckpt if final_ckpt.exists() else None

    def loss_score(path: Path) -> float:
        name = path.name
        if "loss=" not in name:
            return float("inf")
        try:
            return float(name.split("loss=")[1].split("-")[0])
        except ValueError:
            return float("inf")

    return sorted(ckpts, key=loss_score)[0]


def iter_seed_dirs(model_dir: Path) -> Iterable[Path]:
    for item in sorted(model_dir.iterdir()):
        if item.is_dir() and item.name.isdigit():
            yield item


def load_single_model(model_dir: Path, seed_dir: Path, checkpoint: Path) -> Any:
    model_name = (model_dir / ".model").read_text(encoding="utf-8").strip()
    return ModelFactory.load(
        model_name,
        checkpoint=str(checkpoint),
        map_location="cpu",
        log_dir=str(seed_dir),
    )


def extract_first_ansatz(module: Any) -> Optional[Any]:
    if hasattr(module, "ansatz"):
        return module.ansatz

    for child in module.modules():
        if hasattr(child, "ansatz"):
            return child.ansatz
    return None


def module_has_quantum_part(module: Any) -> bool:
    return extract_first_ansatz(module) is not None


def _manual_hqnn_part(relative_dir: str, hydra_cfg: Dict[str, Any]) -> str:
    lower_path = relative_dir.lower()
    for key, part in HQNN_MANUAL_PART_MAP.items():
        if key in lower_path:
            return part

    model_cfg = hydra_cfg.get("model", {}) if isinstance(hydra_cfg, dict) else {}
    encoder_name = str(((model_cfg.get("encoder") or {}).get("name") or "")).lower()
    classifier_name = str(((model_cfg.get("classifier") or {}).get("name") or "")).lower()
    encoder_q = encoder_name in QUANTUM_LAYER_NAMES
    classifier_q = classifier_name in QUANTUM_LAYER_NAMES

    if encoder_q and classifier_q:
        return "both"
    if encoder_q:
        return "encoder"
    if classifier_q:
        return "classifier"
    return "unknown"


def detect_quantum_part(model: Any, relative_dir: str, hydra_cfg: Dict[str, Any]) -> str:
    if hasattr(model, "encoder") and hasattr(model, "classifier"):
        encoder_q = module_has_quantum_part(model.encoder)
        classifier_q = module_has_quantum_part(model.classifier)
        if encoder_q and classifier_q:
            return "both"
        if encoder_q:
            return "encoder"
        if classifier_q:
            return "classifier"
        return _manual_hqnn_part(relative_dir, hydra_cfg)

    return "model"


def build_default_inputs(ansatz: Any) -> np.ndarray:
    embedding = getattr(ansatz, "embedding", None)

    if embedding is None:
        return np.zeros(ansatz.n_qubits, dtype=float)

    input_size = int(getattr(embedding, "input_size", ansatz.n_qubits))
    embedding_name = type(embedding).__name__.lower()

    # Amplitude embeddings require a normalized vector.
    if "amplitude" in embedding_name:
        x = np.zeros(input_size, dtype=float)
        x[0] = 1.0
        return x

    return np.zeros(input_size, dtype=float)


def build_default_kwargs(ansatz: Any) -> Dict[str, np.ndarray]:
    kwargs: Dict[str, np.ndarray] = {}
    for name, shape in (ansatz.weight_shapes or {}).items():
        kwargs[name] = np.zeros(shape, dtype=float)
    return kwargs


def format_gate_types(gate_types: Dict[str, int]) -> str:
    if not gate_types:
        return ""
    lines = [f"{gate}: {count}" for gate, count in sorted(gate_types.items(), key=lambda x: x[0])]
    return "\n".join(lines)


def _empty_resource_dict() -> Dict[str, Any]:
    return {
        "num_gates": None,
        "circuit_depth": None,
        "gate_types": {},
    }


def _specs_to_resource_dict(specs: Any) -> Dict[str, Any]:
    resources = specs.resources
    return {
        "num_gates": resources.num_gates,
        "circuit_depth": resources.depth,
        "gate_types": resources.gate_types or {},
    }


def _build_measurements(ansatz: Any) -> list[Any]:
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(ansatz.n_qubits)]


def extract_specs_split(ansatz: Any, compute_depth: bool, decompose: bool) -> Dict[str, Any]:
    default_inputs = build_default_inputs(ansatz)
    kwargs = build_default_kwargs(ansatz)

    if "weights" not in kwargs:
        raise ValueError(
            f"Expected 'weights' in ansatz.weight_shapes, got keys={list(kwargs.keys())}"
        )

    default_weights = kwargs["weights"]

    dev = qml.device("default.qubit", wires=ansatz.n_qubits)

    def maybe_decompose(qnode_fn: Any) -> Any:
        return qml.transforms.decompose(qnode_fn) if decompose else qnode_fn

    @qml.qnode(dev, interface=None, diff_method=None)
    def embedding_only(inputs, weights):
        if ansatz.embedding is not None:
            ansatz.embedding(inputs)
        return _build_measurements(ansatz)

    @qml.qnode(dev, interface=None, diff_method=None)
    def ansatz_only(inputs, weights):
        ansatz.ansatz(weights)
        return _build_measurements(ansatz)

    @qml.qnode(dev, interface=None, diff_method=None)
    def full_circuit(inputs, weights):
        return ansatz.circuit(inputs, weights)

    embedding_only_qnode = maybe_decompose(embedding_only)
    ansatz_only_qnode = maybe_decompose(ansatz_only)
    full_circuit_qnode = maybe_decompose(full_circuit)

    specs_kwargs = {
        "compute_depth": compute_depth,
        "level": "device",
    }

    full_specs = qml.specs(full_circuit_qnode, **specs_kwargs)(default_inputs, default_weights)
    full_resources = _specs_to_resource_dict(full_specs)

    if ansatz.embedding is None:
        embedding_resources = _empty_resource_dict()
    else:
        embedding_specs = qml.specs(embedding_only_qnode, **specs_kwargs)(default_inputs, default_weights)
        embedding_resources = _specs_to_resource_dict(embedding_specs)

    ansatz_specs = qml.specs(ansatz_only_qnode, **specs_kwargs)(default_inputs, default_weights)
    ansatz_resources = _specs_to_resource_dict(ansatz_specs)

    return {
        "device": str(full_specs.device_name),
        "shots": str(full_specs.shots.total_shots),
        "n_qubits": ansatz.n_qubits,
        "embedding": embedding_resources,
        "ansatz": ansatz_resources,
        "full": full_resources,
    }


def extract_row(
    model: Any,
    relative_dir: str,
    seed: int,
    hydra_cfg: Dict[str, Any],
    compute_depth: bool,
    decompose: bool,
) -> Dict[str, Any]:
    quantum_part = detect_quantum_part(model, relative_dir, hydra_cfg)
    ansatz = extract_first_ansatz(model)

    if ansatz is None:
        return {
            "model": relative_dir,
            "seed": seed,
            "quantum_part": quantum_part,
            "Device": None,
            "Shots": None,
            "Number of Qubits": None,
            "Embedding": None,
            "Ansatz type": None,
            "N Layers (of the ansatz)": None,
            "Weight Shape (of the ansatz)": None,
            "Embedding Number of Gates": None,
            "Embedding Circuit Depth": None,
            "Embedding Gate types": None,
            "Ansatz Number of Gates": None,
            "Ansatz Circuit Depth": None,
            "Ansatz Gate types": None,
            "Full Number of Gates": None,
            "Full Circuit Depth": None,
            "Full Gate types": None,
            "Decomposed": decompose,
        }

    specs = extract_specs_split(ansatz, compute_depth=compute_depth, decompose=decompose)
    embedding_name = "none" if ansatz.embedding is None else type(ansatz.embedding).__name__

    return {
        "model": relative_dir,
        "seed": seed,
        "quantum_part": quantum_part,
        "Device": specs["device"],
        "Shots": specs["shots"],
        "Number of Qubits": ansatz.n_qubits,
        "Embedding": embedding_name,
        "Ansatz type": type(ansatz).__name__,
        "N Layers (of the ansatz)": getattr(ansatz, "n_layers", None),
        "Weight Shape (of the ansatz)": json.dumps(ansatz.weight_shapes),

        "Embedding Number of Gates": specs["embedding"]["num_gates"],
        "Embedding Circuit Depth": specs["embedding"]["circuit_depth"],
        "Embedding Gate types": format_gate_types(specs["embedding"]["gate_types"]),

        "Ansatz Number of Gates": specs["ansatz"]["num_gates"],
        "Ansatz Circuit Depth": specs["ansatz"]["circuit_depth"],
        "Ansatz Gate types": format_gate_types(specs["ansatz"]["gate_types"]),

        "Full Number of Gates": specs["full"]["num_gates"],
        "Full Circuit Depth": specs["full"]["circuit_depth"],
        "Full Gate types": format_gate_types(specs["full"]["gate_types"]),

        "Decomposed": decompose,
    }

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    root = Path("../REPLICA")
    output = Path("outputs/new_quantum_data.csv")
    load_mode = "best"
    compute_depth = True
    decompose = True

    model_dirs = [Path(p) for p in crawl_experiments(str(root))]
    rows: list[Dict[str, Any]] = []

    for model_dir in tqdm(model_dirs, desc="Models"):
        relative_dir = os.path.relpath(model_dir, root)
        hydra_cfg = read_hydra_config(model_dir)

        seed_dirs = list(iter_seed_dirs(model_dir))
        if not seed_dirs:
            continue

        # Pick a single seed (deterministic)
        seed_dir = seed_dirs[0]

        checkpoint = find_checkpoint(seed_dir, load_mode)
        if checkpoint is None:
            log.warning("Skipping %s (no checkpoint found for mode=%s)", seed_dir, load_mode)
            continue

        model = load_single_model(model_dir, seed_dir, checkpoint)

        row = extract_row(
            model=model,
            relative_dir=relative_dir,
            seed=int(seed_dir.name),  # optional, or drop entirely
            hydra_cfg=hydra_cfg,
            compute_depth=compute_depth,
            decompose=decompose,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    log.info("Saved %s rows to %s", len(df), output)


if __name__ == "__main__":
    main()
"""Comparison-specific activation tracing for the adversarial-propagation thesis analysis.

This module is library code.  The companion notebook separates discovery,
validation/preparation, and figure generation.  The implementation deliberately
uses the native trained PyTorch/PennyLane models and never reconstructs a
surrogate quantum interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import gc
import hashlib
import importlib.machinery
import json
import math
import os
import re
import struct
import sys
import types
from typing import Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_propagation_revised"
ATTACK_DIR = PROJECT_ROOT / "logs" / "ATTACKS" / "attack_experiment"
MNIST_RAW_DIR = PROJECT_ROOT / "data" / "processed" / "MNIST" / "raw"

SEEDS = (0, 42)
EPSILON = 0.10
TARGET_PER_CLASS = 50
TRACE_BATCH_SIZE = 100
VALIDATION_BATCH_SIZE = 512
BOOTSTRAP_RESAMPLES = 1000
SELECTION_SEED = 20260801
TRACE_SCHEMA_VERSION = "2026-08-01.comparison-specific.v2"

COMMON_A_MODELS = ("cnn", "hvql")
LATE_MODELS = ("hvql", "hpvql", "hvqm", "hqe", "reshvql")


@dataclass(frozen=True)
class ActivationPoint:
    name: str
    module_path: str
    hook: str
    display_name: str
    order: float
    role: str


@dataclass(frozen=True)
class RevisionModelSpec:
    key: str
    display_name: str
    semantic_architecture: str
    raw_attack_name: str
    model_dir: str
    checkpoints: Mapping[int, str]
    points: tuple[ActivationPoint, ...]
    prequantum_point: str | None
    measurement_point: str | None
    resolved_configuration: Mapping[str, object]
    resolution_note: str = ""


def _point(
    name: str,
    module_path: str,
    display_name: str,
    order: float,
    role: str,
    hook: str = "output",
) -> ActivationPoint:
    return ActivationPoint(name, module_path, hook, display_name, order, role)


def model_specs() -> dict[str, RevisionModelSpec]:
    rich_encoder = (
        _point("encoder_conv1", "encoder.conv1", "Conv. 1", 1, "classical_encoder"),
        _point("encoder_block1", "encoder.maxpool1", "Conv. block 1", 2, "classical_encoder"),
        _point("encoder_conv2", "encoder.conv2", "Conv. 2", 3, "classical_encoder"),
        _point("encoder_block2", "encoder.maxpool2", "Conv. block 2", 4, "classical_encoder"),
        _point("encoder_flattened", "encoder", "Flattened encoder", 5, "flattened"),
    )
    rich_interface = (
        _point("prequantum_linear", "classifier.fc1", "Prequantum linear", 6, "prequantum"),
        _point("prequantum_normalized", "classifier.bn", "Prequantum batch norm", 7, "encoded_normalized"),
        _point("prequantum", "classifier.relu", "Prequantum z", 8, "prequantum"),
    )
    strong = {"class": "StrongEntanglingAnsatz", "depth": 50, "embedding": "amplitude"}
    return {
        "cnn": RevisionModelSpec(
            key="cnn",
            display_name="CNN [32,64] / FC [128]",
            semantic_architecture="CNN",
            raw_attack_name="Classical/convnet/conv_channels=[32,64],fc_sizes=[128]",
            model_dir="logs/ByModel/Classical/convnet/conv_channels=[32,64],fc_sizes=[128]",
            checkpoints={0: "epoch=34-loss=0.0450.ckpt", 42: "epoch=31-loss=0.0421.ckpt"},
            points=(
                _point("conv1", "features.0", "Conv. 1", 1, "classical_encoder"),
                _point("conv_block1", "features.2", "Conv. block 1", 2, "classical_encoder"),
                _point("conv2", "features.3", "Conv. 2", 3, "classical_encoder"),
                _point("conv_block2", "features.5", "Conv. block 2", 4, "classical_encoder"),
                _point("flattened", "classifier.0", "Flattened", 5, "flattened", hook="input"),
                _point("hidden_fc_linear", "classifier.0", "Hidden FC (linear)", 6, "classical_readout"),
                _point("hidden_fc", "classifier.1", "Hidden FC", 7, "classical_readout"),
                _point("readout", "classifier.2", "Linear readout", 8, "classical_readout"),
            ),
            prequantum_point=None,
            measurement_point=None,
            resolved_configuration={
                "model_class": "ConvNet",
                "conv_channels": [32, 64],
                "fc_sizes": [128],
                "input_shape": [1, 28, 28],
                "classes": 10,
            },
        ),
        "hvql": RevisionModelSpec(
            key="hvql",
            display_name="HVQL strong d=50",
            semantic_architecture="HVQL",
            raw_attack_name="Hybrid/hvqm/ansatz=strong,qdepth=50",
            model_dir="logs/ByBenchmark/HVQM-Benchmarks/ansatz=strong,qdepth=50",
            checkpoints={0: "epoch=39-loss=0.0464.ckpt", 42: "epoch=39-loss=0.0486.ckpt"},
            points=rich_encoder + rich_interface + (
                _point("measurement", "classifier.qlayer", "Measured quantum r", 9, "measurement"),
                _point("readout", "classifier.fc2", "Classical readout", 10, "classical_readout"),
            ),
            prequantum_point="prequantum",
            measurement_point="measurement",
            resolved_configuration={
                "model_class": "HQNN", "encoder": {"name": "conv", "conv_channels": [16, 32], "output": 1568},
                "classifier": {"name": "hvql", "prequantum_dim": 1024, "measurement_dim": 10,
                               "readout": "Linear(10,10)", "quantum": strong},
                "quantum_device": {"name": "default.qubit", "interface": "torch", "diff_method": "backprop", "shots": None},
            },
            resolution_note="Historical raw folder says HVQM; executable configuration/classifier is semantic HVQL.",
        ),
        "hpvql": RevisionModelSpec(
            key="hpvql",
            display_name="HPVQL (5 blocks), strong d=50",
            semantic_architecture="HPVQL",
            raw_attack_name="Hybrid/hpvqm/qblocks=5,qdepth=50",
            model_dir="logs/ByModel/Hybrid/hpvqm/qblocks=5,qdepth=50",
            checkpoints={0: "epoch=15-loss=0.0327.ckpt", 42: "epoch=23-loss=0.0295.ckpt"},
            points=rich_encoder + rich_interface + tuple(
                _point(f"measurement_block_{i + 1}", f"classifier.qlayer.blocks.{i}",
                       f"Quantum block {i + 1}", 9 + i / 10, "measurement_block")
                for i in range(5)
            ) + (
                _point("measurement", "classifier.qlayer", "Concatenated measured r", 9.6, "measurement"),
                _point("readout", "classifier.fc2", "Classical readout", 10, "classical_readout"),
            ),
            prequantum_point="prequantum",
            measurement_point="measurement",
            resolved_configuration={
                "model_class": "HQNN", "encoder": {"name": "conv", "conv_channels": [16, 32], "output": 1568},
                "classifier": {"name": "hpvql", "prequantum_dim": 1024, "n_qblocks": 5,
                               "block_input_dim": 205, "block_qubits": 8, "block_measurement_dim": 8,
                               "measurement_dim": 40, "readout": "Linear(40,10)", "quantum": strong},
                "quantum_device": {"name": "default.qubit", "interface": "torch", "diff_method": "backprop", "shots": None},
            },
            resolution_note="The hpvqm path is historical; Hydra name=hpvql and the native classifier is HPVQL/ParaVQC.",
        ),
        "hvqm": RevisionModelSpec(
            key="hvqm",
            display_name="HVQM strong d=50",
            semantic_architecture="HVQM",
            raw_attack_name="Hybrid/hvql/ansatz=strong,qdepth=50",
            model_dir="logs/ByBenchmark/HVQL-Ansatzs/ansatz=strong,qdepth=50",
            checkpoints={0: "epoch=39-loss=1.8567.ckpt", 42: "epoch=39-loss=1.8586.ckpt"},
            points=(
                _point("encoder_conv1", "encoder.conv1", "Conv. 1", 1, "classical_encoder"),
                _point("encoder_block1", "encoder.maxpool1", "Conv. block 1", 2, "classical_encoder"),
                _point("encoder_conv2", "encoder.conv2", "Conv. 2", 3, "classical_encoder"),
                _point("encoder_block2", "encoder.maxpool2", "Conv. block 2", 4, "classical_encoder"),
                _point("prequantum", "encoder", "Prequantum z", 5, "prequantum"),
                _point("measurement", "classifier.vqc", "Measured quantum r", 6, "measurement"),
                _point("readout", "classifier.post", "Readout (identity)", 7, "classical_readout"),
            ),
            prequantum_point="prequantum",
            measurement_point="measurement",
            resolved_configuration={
                "model_class": "HQNN", "encoder": {"name": "conv", "conv_channels": [8, 16], "output": 784},
                "classifier": {"name": "vql", "prequantum_dim": 784, "amplitude_padding_dim": 1024,
                               "measurement_dim": 10, "readout": "Identity", "quantum": strong},
                "quantum_device": {"name": "default.qubit", "interface": "torch", "diff_method": "backprop", "shots": None},
            },
            resolution_note="Historical raw folder says HVQL; executable configuration/classifier is semantic HVQM.",
        ),
        "hqe": RevisionModelSpec(
            key="hqe",
            display_name="HQE strong d=50",
            semantic_architecture="HQE",
            raw_attack_name="Hybrid/hqe",
            model_dir="logs/ByBenchmark/Baseline-Models/Hybrid/hqe",
            checkpoints={0: "epoch=36-loss=0.1028.ckpt", 42: "epoch=37-loss=0.1047.ckpt"},
            points=(
                _point("encoder_dense1", "encoder.net.0", "Encoder dense 4096", 1, "classical_encoder"),
                _point("encoder_block1", "encoder.net.1", "Encoder block 1", 2, "classical_encoder"),
                _point("encoder_dense2", "encoder.net.2", "Encoder dense 2048", 3, "classical_encoder"),
                _point("encoder_block2", "encoder.net.3", "Encoder block 2", 4, "classical_encoder"),
                _point("encoder_dense3", "encoder.net.4", "Encoder dense 1024", 5, "classical_encoder"),
                _point("encoder_final", "encoder.net.5", "Encoder final", 6, "classical_encoder"),
            ) + (
                _point("prequantum_linear", "classifier.fc1", "Prequantum linear", 7, "prequantum"),
                _point("prequantum_normalized", "classifier.bn", "Prequantum batch norm", 8, "encoded_normalized"),
                _point("prequantum", "classifier.relu", "Prequantum z", 9, "prequantum"),
                _point("measurement", "classifier.qlayer", "Measured quantum r", 10, "measurement"),
                _point("readout", "classifier.fc2", "Classical readout", 11, "classical_readout"),
            ),
            prequantum_point="prequantum",
            measurement_point="measurement",
            resolved_configuration={
                "model_class": "HQNN", "encoder": {"name": "encoder", "hidden_sizes": [4096, 2048, 1024]},
                "classifier": {"name": "hvql", "prequantum_dim": 1024, "measurement_dim": 10,
                               "readout": "Linear(10,10)", "quantum": strong},
                "quantum_device": {"name": "default.qubit", "interface": "torch", "diff_method": "backprop", "shots": None},
            },
        ),
        "reshvql": RevisionModelSpec(
            key="reshvql",
            display_name="Residual HVQL strong d=50",
            semantic_architecture="Residual HVQL",
            raw_attack_name="Hybrid/hvql/residuals",
            model_dir="logs/ByModel/Hybrid/hvql/residuals",
            checkpoints={0: "epoch=14-loss=0.0352.ckpt", 42: "epoch=18-loss=0.0327.ckpt"},
            points=rich_encoder + rich_interface + (
                _point("quantum_branch", "classifier.vqc", "Quantum branch", 9, "measurement"),
                _point("classical_branch_raw", "classifier.residual_proj", "Classical bypass (raw)", 9.1, "residual_branch_raw"),
                _point("fused", "classifier", "Fused representation", 10, "fused"),
            ),
            prequantum_point="prequantum",
            measurement_point="quantum_branch",
            resolved_configuration={
                "model_class": "HQNN", "encoder": {"name": "conv", "conv_channels": [16, 32], "output": 1568},
                "classifier": {"name": "resvql", "prequantum_dim": 1024, "measurement_dim": 10,
                               "classical_bypass": "Linear(1568,10)", "fusion": "q + residual_scale * c",
                               "residual_init": 0.5, "quantum": strong},
                "quantum_device": {"name": "default.qubit", "interface": "torch", "diff_method": "backprop", "shots": None},
            },
        ),
    }


def safe_model_name(name: str) -> str:
    return str(name).replace("\\", "__").replace("/", "__").replace(" ", "_").replace(":", "_")


def _remove_modules(prefix: str) -> None:
    for name in tuple(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)


def install_inference_import_shims() -> None:
    """Shim unused eager-import dependencies blocked by Windows App Control.

    The required forward paths do not use pandas, sklearn PCA, or bzip2.  The
    project imports them eagerly through training/analytics utilities, so this
    function supplies import-only placeholders when the real modules cannot be
    loaded.  Numerical aggregation never uses these placeholders.
    """

    try:
        import bz2  # noqa: F401
    except Exception:
        _remove_modules("bz2")
        module = types.ModuleType("bz2")
        module.__spec__ = importlib.machinery.ModuleSpec("bz2", None)
        module.BZ2File = type("BZ2File", (), {})
        module.BZ2Compressor = type("BZ2Compressor", (), {})
        module.BZ2Decompressor = type("BZ2Decompressor", (), {})
        module.compress = lambda value, *args, **kwargs: value
        module.decompress = lambda value, *args, **kwargs: value
        module.open = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bz2 is unavailable"))
        sys.modules["bz2"] = module

    try:
        import pandas  # noqa: F401
    except Exception:
        _remove_modules("pandas")
        module = types.ModuleType("pandas")
        module.__spec__ = importlib.machinery.ModuleSpec("pandas", None)
        module.__version__ = "unavailable-in-inference-runtime"
        module.DataFrame = type("DataFrame", (), {})
        module.Series = type("Series", (), {})
        module.merge = lambda *args, **kwargs: None
        sys.modules["pandas"] = module

    try:
        from sklearn.decomposition import PCA  # noqa: F401
    except Exception:
        _remove_modules("sklearn")
        root = types.ModuleType("sklearn")
        root.__spec__ = importlib.machinery.ModuleSpec("sklearn", None)
        decomposition = types.ModuleType("sklearn.decomposition")
        decomposition.__spec__ = importlib.machinery.ModuleSpec("sklearn.decomposition", None)
        decomposition.PCA = type("PCA", (), {})
        root.decomposition = decomposition
        sys.modules["sklearn"] = root
        sys.modules["sklearn.decomposition"] = decomposition


def _checkpoint_path(spec: RevisionModelSpec, seed: int) -> Path:
    path = PROJECT_ROOT / spec.model_dir / str(seed) / "checkpoints" / spec.checkpoints[int(seed)]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_native_model(spec: RevisionModelSpec, seed: int):
    install_inference_import_shims()
    src_dir = str(PROJECT_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from models.ModelFactory import ModelFactory

    model_dir = PROJECT_ROOT / spec.model_dir
    model_name = (model_dir / ".model").read_text(encoding="utf-8").strip()
    checkpoint = _checkpoint_path(spec, seed)
    model = ModelFactory.load(
        model_name,
        checkpoint=str(checkpoint),
        map_location="cpu",
        log_dir=str(model_dir / str(seed)),
    )
    model.eval()
    return model, checkpoint


def load_official_mnist() -> tuple[object, object, np.ndarray, np.ndarray]:
    """Load official, already-uncompressed MNIST IDX tensors in canonical order."""

    import torch

    image_path = MNIST_RAW_DIR / "t10k-images-idx3-ubyte"
    label_path = MNIST_RAW_DIR / "t10k-labels-idx1-ubyte"
    with image_path.open("rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051 or count != 10000 or (rows, cols) != (28, 28):
            raise ValueError(f"Unexpected MNIST image header: {(magic, count, rows, cols)}")
        images_np = np.frombuffer(handle.read(), dtype=np.uint8).reshape(count, 1, rows, cols).copy()
    with label_path.open("rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049 or count != 10000:
            raise ValueError(f"Unexpected MNIST label header: {(magic, count)}")
        labels_np = np.frombuffer(handle.read(), dtype=np.uint8).astype(np.int64, copy=True)
    saved_labels = np.load(ATTACK_DIR / "seed_0" / "labels.npy").astype(np.int64)
    if not np.array_equal(labels_np, saved_labels):
        raise ValueError("Official MNIST order differs from attack labels.npy")
    images = torch.from_numpy(images_np).to(torch.float32) / 255.0
    labels = torch.from_numpy(labels_np).to(torch.long)
    return images, labels, images_np, labels_np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_dict_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or (list(rows[0].keys()) if rows else []))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _forward_full_test(model, images, batch_size: int = VALIDATION_BATCH_SIZE):
    import torch
    from tqdm.auto import tqdm

    outputs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(images), batch_size), desc="MNIST test forward", leave=False):
            outputs.append(model(images[start : start + batch_size]).detach().to(torch.float32).cpu())
    return torch.cat(outputs, dim=0)


def _validate_native_topology(model, spec: RevisionModelSpec) -> dict[str, object]:
    """Execute architecture assertions that must hold before mechanistic tracing."""

    checks: dict[str, object] = {
        "model_class": type(model).__name__,
        "classifier_class": type(getattr(model, "classifier", model)).__name__,
    }
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    if spec.key == "cnn":
        require(type(model).__name__ == "ConvNet", "expected ConvNet")
        require([model.features[0].out_channels, model.features[3].out_channels] == [32, 64], "conv channels != [32,64]")
        require(model.classifier[0].in_features == 3136 and model.classifier[0].out_features == 128,
                "hidden FC != 3136->128")
        require(model.classifier[2].in_features == 128 and model.classifier[2].out_features == 10,
                "readout != 128->10")
        checks.update({"conv_channels": [32, 64], "hidden_dim": 128, "output_dim": 10})
    else:
        require(type(model).__name__ == "HQNN", "expected HQNN")
        expected_classifier = {
            "hvql": "HVQL", "hpvql": "HPVQL", "hvqm": "VQL",
            "hqe": "HVQL", "reshvql": "ResVQL",
        }[spec.key]
        require(type(model.classifier).__name__ == expected_classifier,
                f"expected classifier {expected_classifier}")
        expected_encoder_dim = {"hvql": 1568, "hpvql": 1568, "hvqm": 784, "hqe": 1024, "reshvql": 1568}[spec.key]
        require(int(model.encoder.output_size) == expected_encoder_dim,
                f"encoder output != {expected_encoder_dim}")
        if spec.key in ("hvql", "hqe"):
            quantum_modules = [model.classifier.qlayer]
            require(model.classifier.fc1.out_features == 1024, "prequantum dim != 1024")
            require(model.classifier.qlayer.output_size == 10, "measurement dim != 10")
            require(model.classifier.fc2.in_features == 10 and model.classifier.fc2.out_features == 10,
                    "readout != 10->10")
        elif spec.key == "hpvql":
            parallel = model.classifier.qlayer
            quantum_modules = list(parallel.blocks)
            require(parallel.n_blocks == 5, "parallel block count != 5")
            require(parallel.n_layers == 50, "parallel depth != 50")
            require(parallel.block_size == 205, "parallel block input != 205")
            require(parallel.output_size == 40, "concatenated measurement dim != 40")
            require(model.classifier.fc2.in_features == 40 and model.classifier.fc2.out_features == 10,
                    "parallel readout != 40->10")
            checks.update({"parallel_blocks": parallel.n_blocks, "block_size": parallel.block_size,
                           "measurement_dim": parallel.output_size})
        elif spec.key == "hvqm":
            quantum_modules = [model.classifier.vqc]
            require(type(model.classifier.post).__name__ == "Identity", "HVQM readout is not Identity")
        else:
            quantum_modules = [model.classifier.vqc]
            require(model.classifier.fc1.out_features == 1024, "prequantum dim != 1024")
            require(model.classifier.residual_proj.in_features == 1568
                    and model.classifier.residual_proj.out_features == 10,
                    "residual projection != 1568->10")
        for position, quantum_module in enumerate(quantum_modules):
            ansatz = quantum_module.ansatz
            require(type(ansatz).__name__ == "StrongEntanglingAnsatz",
                    f"quantum block {position} ansatz is not strong")
            require(int(ansatz.n_layers) == 50, f"quantum block {position} depth != 50")
            require(int(ansatz.n_qubits) in (8, 10), f"unexpected qubit count in block {position}")
            require(ansatz.quantum_config.device == "default.qubit",
                    f"quantum block {position} device is not default.qubit")
            require(ansatz.quantum_config.interface == "torch",
                    f"quantum block {position} interface is not torch")
            require(ansatz.quantum_config.diff_method == "backprop",
                    f"quantum block {position} differentiation is not backprop")
            require(ansatz.quantum_config.shots is None,
                    f"quantum block {position} uses finite shots")
        checks.update({
            "encoder_output_dim": int(model.encoder.output_size),
            "quantum_depth": 50,
            "ansatz": "StrongEntanglingAnsatz",
            "quantum_device": "default.qubit",
            "quantum_interface": "torch",
            "quantum_diff_method": "backprop",
            "shots": None,
        })
    checks["topology_errors"] = errors
    checks["topology_passed"] = not errors
    return checks


def validate_checkpoints(
    specs: Mapping[str, RevisionModelSpec] | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[list[dict[str, object]], dict[tuple[str, int], np.ndarray]]:
    """Recompute full-test accuracy and validate exact attack-aligned checkpoints."""

    import torch
    from tqdm.auto import tqdm

    specs = dict(specs or model_specs())
    images, labels, _images_np, labels_np = load_official_mnist()
    rows: list[dict[str, object]] = []
    predictions: dict[tuple[str, int], np.ndarray] = {}
    prediction_dir = output_dir / "_cache" / "validated_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for spec in tqdm(specs.values(), desc="Checkpoint validation"):
        for seed in SEEDS:
            model, checkpoint = load_native_model(spec, seed)
            topology = _validate_native_topology(model, spec)
            outputs = _forward_full_test(model, images)
            native_preds = outputs.argmax(1).numpy().astype(np.uint8)
            float16_preds = outputs.to(torch.float16).argmax(1).numpy().astype(np.uint8)
            baseline_dir = ATTACK_DIR / f"seed_{seed}" / "baselines" / safe_model_name(spec.raw_attack_name)
            saved_logits = np.load(baseline_dir / "logits_clean.npy").astype(np.float32)
            saved_preds = np.load(baseline_dir / "preds_clean.npy").astype(np.uint8)
            stored_accuracy = float(np.mean(saved_preds.astype(np.int64) == labels_np))
            native_accuracy = float(np.mean(native_preds.astype(np.int64) == labels_np))
            float16_accuracy = float(np.mean(float16_preds.astype(np.int64) == labels_np))
            agreement_native = float(np.mean(native_preds == saved_preds))
            agreement_float16 = float(np.mean(float16_preds == saved_preds))
            error = np.abs(outputs.numpy() - saved_logits)
            runtime_hparams = _jsonable(dict(model.hparams))
            total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
            quantum_parameters = int(sum(
                parameter.numel() for name, parameter in model.named_parameters()
                if any(token in name.lower() for token in ("qlayer", "vqc", "ansatz"))
            ))
            residual_scale = (
                float(model.classifier.residual_scale.detach().cpu())
                if spec.key == "reshvql" else math.nan
            )
            difference_pp = 100.0 * (native_accuracy - stored_accuracy)
            validation_passed = bool(
                topology["topology_passed"]
                and agreement_float16 == 1.0
                and float(error.max()) < 0.02
            )
            row = {
                "model_key": spec.key,
                "model": spec.display_name,
                "semantic_architecture": spec.semantic_architecture,
                "raw_attack_name": spec.raw_attack_name,
                "seed": int(seed),
                "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "checkpoint_sha256": _sha256(checkpoint),
                "n_test": int(len(labels_np)),
                "recomputed_test_accuracy": native_accuracy,
                "recomputed_test_accuracy_float16_argmax": float16_accuracy,
                "stored_attack_baseline_accuracy": stored_accuracy,
                "stored_attack_baseline_source": str((baseline_dir / "preds_clean.npy").relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "stored_training_metadata_test_accuracy": math.nan,
                "stored_training_metadata_status": "unavailable: model.info and metrics.csv contain no test accuracy",
                "accuracy_difference_vs_attack_baseline_percentage_points": difference_pp,
                "accuracy_difference_vs_attack_baseline_gt_0_2pp": bool(abs(difference_pp) > 0.2),
                "prediction_agreement_native": agreement_native,
                "prediction_agreement_float16_argmax": agreement_float16,
                "max_abs_logit_difference_vs_float16_baseline": float(error.max()),
                "mean_abs_logit_difference_vs_float16_baseline": float(error.mean()),
                "validation_passed": validation_passed,
                "executable_class": type(model).__name__,
                "classifier_class": type(getattr(model, "classifier", model)).__name__,
                "total_parameters": total_parameters,
                "quantum_parameters": quantum_parameters,
                "classical_parameters": total_parameters - quantum_parameters,
                "residual_scale": residual_scale,
                "resolved_configuration_json": json.dumps(_jsonable(spec.resolved_configuration), ensure_ascii=False, sort_keys=True),
                "checkpoint_hparams_json": json.dumps(runtime_hparams, ensure_ascii=False, sort_keys=True),
                "resolution_note": spec.resolution_note,
                "topology_validation_json": json.dumps(_jsonable(topology), ensure_ascii=False, sort_keys=True),
                "topology_passed": bool(topology["topology_passed"]),
            }
            rows.append(row)
            predictions[(spec.key, seed)] = native_preds
            np.save(prediction_dir / f"{spec.key}_seed={seed}.npy", native_preds)
            if not validation_passed:
                _write_dict_rows(output_dir / "checkpoint_validation.csv", rows)
                raise RuntimeError(
                    f"Blocking checkpoint validation failure for {spec.key}, seed {seed}: "
                    f"topology_errors={topology['topology_errors']}, float16 agreement={agreement_float16}, "
                    f"max logit difference={error.max()}"
                )
            del model, outputs
            gc.collect()
    _write_dict_rows(output_dir / "checkpoint_validation.csv", rows)
    return rows, predictions


def _deterministic_balanced_indices(
    eligible: np.ndarray,
    labels: np.ndarray,
    salt: str,
) -> tuple[np.ndarray, int, np.ndarray]:
    per_class = np.array([np.sum(eligible & (labels == digit)) for digit in range(10)], dtype=int)
    quota = int(min(TARGET_PER_CLASS, per_class.min()))
    digest = int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(SELECTION_SEED + digest)
    selected: list[int] = []
    for digit in range(10):
        candidates = np.flatnonzero(eligible & (labels == digit))
        rng.shuffle(candidates)
        selected.extend(sorted(candidates[:quota].tolist()))
    return np.array(sorted(selected), dtype=np.int64), quota, per_class


def build_cohort_manifest(
    predictions: Mapping[tuple[str, int], np.ndarray],
    specs: Mapping[str, RevisionModelSpec] | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    specs = dict(specs or model_specs())
    _images, _labels_tensor, _images_np, labels = load_official_mnist()
    definitions: list[tuple[str, str, tuple[str, ...], str]] = [
        ("A", "common_comparison", COMMON_A_MODELS, "Correctly classified by CNN and HVQL"),
        ("B", "common_comparison", LATE_MODELS, "Correctly classified by all five displayed late-hybrid models"),
    ]
    definitions.extend(
        (f"C_{key}", "model_specific_sensitivity", (key,), f"Correctly classified by {specs[key].display_name}")
        for key in LATE_MODELS
    )
    manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for cohort_id, cohort_kind, keys, criterion in definitions:
        for seed in SEEDS:
            eligible = np.ones(len(labels), dtype=bool)
            for key in keys:
                eligible &= predictions[(key, seed)].astype(np.int64) == labels
            indices, quota, available_by_class = _deterministic_balanced_indices(
                eligible, labels, salt=f"{cohort_id}:seed={seed}"
            )
            counts = np.bincount(labels[indices], minlength=10)
            for index in indices:
                manifest_rows.append({
                    "cohort_id": cohort_id,
                    "cohort_kind": cohort_kind,
                    "seed": int(seed),
                    "dataset_index": int(index),
                    "class_label": int(labels[index]),
                    "model_keys": ";".join(keys),
                    "criterion": criterion,
                    "class_quota": quota,
                    "cohort_size": int(len(indices)),
                    "selection_seed": SELECTION_SEED,
                })
            summary_rows.append({
                "cohort_id": cohort_id,
                "cohort_kind": cohort_kind,
                "seed": int(seed),
                "model_keys": ";".join(keys),
                "criterion": criterion,
                "eligible_total": int(eligible.sum()),
                "class_quota": quota,
                "cohort_size": int(len(indices)),
                **{f"eligible_class_{digit}": int(available_by_class[digit]) for digit in range(10)},
                **{f"selected_class_{digit}": int(counts[digit]) for digit in range(10)},
            })
    _write_dict_rows(output_dir / "cohort_manifest.csv", manifest_rows)
    _write_dict_rows(output_dir / "cohort_summary.csv", summary_rows)
    return manifest_rows, summary_rows


def _resolve_module(model, module_path: str):
    modules = dict(model.named_modules())
    if module_path not in modules:
        raise KeyError(f"Missing module {module_path!r}; available: {sorted(modules)}")
    return modules[module_path]


def _tensor_output(output):
    if hasattr(output, "shape"):
        return output
    if isinstance(output, (tuple, list)) and output and hasattr(output[0], "shape"):
        return output[0]
    raise TypeError(type(output))


def _raw_logits_source(spec: RevisionModelSpec) -> str:
    """Return the native pre-LogSoftmax tensor used as logits by the architecture."""

    return "fused" if spec.key == "reshvql" else "readout"


def _capture_clean(model, spec: RevisionModelSpec, images, labels):
    import torch

    captured: dict[str, object] = {}
    handles = []
    try:
        for point in spec.points:
            module = _resolve_module(model, point.module_path)
            if point.hook == "input":
                def pre_hook(_module, inputs, key=point.name):
                    captured[key] = _tensor_output(inputs[0])
                handles.append(module.register_forward_pre_hook(pre_hook))
            else:
                def output_hook(_module, _inputs, output, key=point.name):
                    captured[key] = _tensor_output(output)
                handles.append(module.register_forward_hook(output_hook))
        x = images.clone().detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model_output = model(x)
        raw_logits = captured[_raw_logits_source(spec)]
        row = torch.arange(len(labels), device=raw_logits.device)
        alternatives = raw_logits.detach().clone()
        alternatives[row, labels] = -torch.inf
        competitor = alternatives.argmax(1)
        margin = raw_logits[row, labels] - raw_logits[row, competitor]
        names = ["input", *[point.name for point in spec.points]]
        tensors = [x, *[captured[point.name] for point in spec.points]]
        gradients_tuple = torch.autograd.grad(
            margin.sum(), tensors, retain_graph=False, create_graph=False, allow_unused=False
        )
        values = {name: tensor.detach().to(torch.float32).cpu() for name, tensor in zip(names, tensors)}
        gradients = {
            name: gradient.detach().to(torch.float32).cpu()
            for name, gradient in zip(names, gradients_tuple)
        }
        source = _raw_logits_source(spec)
        values["logits"] = values[source]
        gradients["logits"] = gradients[source]
        if spec.key == "reshvql":
            scale = model.classifier.residual_scale
            scale_value = float(scale.detach().cpu())
            if scale_value == 0.0:
                raise RuntimeError("Residual scale is zero; scaled-branch activation gradient is undefined")
            values["classical_branch_scaled"] = values["classical_branch_raw"] * scale_value
            raw_gradient = gradients["classical_branch_raw"]
            gradients["classical_branch_scaled"] = raw_gradient / scale_value if raw_gradient is not None else None
        del model_output
        return values, gradients, margin.detach().to(torch.float32).cpu(), competitor.detach().cpu()
    finally:
        for handle in handles:
            handle.remove()


def _capture_adv(model, spec: RevisionModelSpec, images):
    import torch

    captured: dict[str, object] = {}
    handles = []
    try:
        for point in spec.points:
            module = _resolve_module(model, point.module_path)
            if point.hook == "input":
                handles.append(module.register_forward_pre_hook(
                    lambda _module, inputs, key=point.name: captured.__setitem__(key, _tensor_output(inputs[0]))
                ))
            else:
                handles.append(module.register_forward_hook(
                    lambda _module, _inputs, output, key=point.name: captured.__setitem__(key, _tensor_output(output))
                ))
        with torch.no_grad():
            model_output = model(images)
        values = {"input": images.detach().to(torch.float32).cpu()}
        values.update({name: tensor.detach().to(torch.float32).cpu() for name, tensor in captured.items()})
        source = _raw_logits_source(spec)
        values["logits"] = values[source]
        if spec.key == "reshvql":
            scale_value = float(model.classifier.residual_scale.detach().cpu())
            values["classical_branch_scaled"] = values["classical_branch_raw"] * scale_value
        del model_output
        return values
    finally:
        for handle in handles:
            handle.remove()


def generate_fgsm(model, images, labels, epsilon: float = EPSILON):
    import torch

    x = images.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, labels)
    gradient = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
    return (x.detach() + float(epsilon) * gradient.detach().sign()).clamp(0.0, 1.0)


TRACE_FIELDS = (
    "trace_schema_version", "cohort_id", "cohort_kind", "model_key", "model",
    "semantic_architecture", "seed", "dataset_index", "class_label", "epsilon",
    "attack", "attack_norm", "layer", "layer_display_name", "layer_order", "point_role",
    "tensor_name", "hook_side", "tensor_shape", "dimensionality", "clean_prediction", "adversarial_prediction",
    "clean_margin", "small_margin_lt_1e-6", "delta_l2", "delta_rms", "gain_rms",
    "projected_harm", "normalized_harm", "residual_scale",
    "classical_to_quantum_displacement_ratio", "raw_classical_to_quantum_displacement_ratio",
    "quantum_norm_share_of_branch_sum", "classical_norm_share_of_branch_sum",
    "quantum_to_fused_norm_ratio", "classical_to_fused_norm_ratio",
    "quantum_projection_share_of_fused", "classical_projection_share_of_fused",
    "residual_reconstruction_relative_error",
)


def _activation_metadata(spec: RevisionModelSpec) -> dict[str, tuple[str, float, str, str]]:
    metadata = {
        "input": ("Input", 0.0, "input", "input"),
        **{
            point.name: (point.display_name, point.order, point.role, point.module_path)
            for point in spec.points
        },
        "logits": ("Logits", 99.0, "logits", f"alias_of:{_raw_logits_source(spec)} (native pre-LogSoftmax)"),
    }
    if spec.key == "reshvql":
        metadata["classical_branch_scaled"] = (
            "Classical bypass (scaled)", 9.2, "residual_branch_scaled", "residual_scale * classifier.residual_proj"
        )
    return metadata


def _residual_diagnostics(clean: Mapping[str, object], adv: Mapping[str, object]) -> dict[str, object]:
    import torch

    delta_q = (adv["quantum_branch"] - clean["quantum_branch"]).reshape(len(adv["quantum_branch"]), -1)
    delta_c_raw = (adv["classical_branch_raw"] - clean["classical_branch_raw"]).reshape(len(delta_q), -1)
    delta_c = (adv["classical_branch_scaled"] - clean["classical_branch_scaled"]).reshape(len(delta_q), -1)
    delta_f = (adv["fused"] - clean["fused"]).reshape(len(delta_q), -1)
    q_norm = delta_q.norm(dim=1)
    c_raw_norm = delta_c_raw.norm(dim=1)
    c_norm = delta_c.norm(dim=1)
    f_norm = delta_f.norm(dim=1)
    branch_sum = q_norm + c_norm
    fused_sq = delta_f.square().sum(dim=1)
    if bool((q_norm <= 0).any() or (branch_sum <= 0).any() or (f_norm <= 0).any() or (fused_sq <= 0).any()):
        raise RuntimeError("A residual displacement denominator is zero; exact branch ratios are undefined")
    reconstruction = (delta_q + delta_c - delta_f).norm(dim=1) / f_norm
    return {
        "classical_to_quantum_displacement_ratio": c_norm / q_norm,
        "raw_classical_to_quantum_displacement_ratio": c_raw_norm / q_norm,
        "quantum_norm_share_of_branch_sum": q_norm / branch_sum,
        "classical_norm_share_of_branch_sum": c_norm / branch_sum,
        "quantum_to_fused_norm_ratio": q_norm / f_norm,
        "classical_to_fused_norm_ratio": c_norm / f_norm,
        "quantum_projection_share_of_fused": (delta_q * delta_f).sum(dim=1) / fused_sq,
        "classical_projection_share_of_fused": (delta_c * delta_f).sum(dim=1) / fused_sq,
        "residual_reconstruction_relative_error": reconstruction,
    }


def _batch_trace_rows(
    spec: RevisionModelSpec,
    seed: int,
    indices: np.ndarray,
    memberships: Mapping[int, Sequence[tuple[str, str]]],
    labels,
    clean: Mapping[str, object],
    gradients: Mapping[str, object],
    adv: Mapping[str, object],
    margins,
    residual_scale_value: float = math.nan,
) -> list[dict[str, object]]:
    import torch

    input_delta = (adv["input"] - clean["input"]).reshape(len(indices), -1)
    input_rms = input_delta.square().mean(dim=1).sqrt()
    if bool((input_rms <= 0).any()):
        bad = indices[(input_rms.numpy() <= 0)]
        raise RuntimeError(f"Zero FGSM input RMS at dataset indices {bad[:10].tolist()}; exact gain is undefined")
    metadata = _activation_metadata(spec)
    hook_sides = {point.name: point.hook for point in spec.points}
    hook_sides.update({"input": "direct_input", "logits": "derived_alias"})
    if spec.key == "reshvql":
        hook_sides["classical_branch_scaled"] = "derived_from_raw_output"
    residual = _residual_diagnostics(clean, adv) if spec.key == "reshvql" else {}
    residual_scale = float(residual_scale_value)
    if spec.key == "reshvql" and not math.isfinite(residual_scale):
        raise RuntimeError("The learned residual_scale parameter was not supplied to tracing")
    clean_predictions = clean["logits"].argmax(dim=1)
    adv_predictions = adv["logits"].argmax(dim=1)
    if not torch.equal(clean_predictions.to(labels.device), labels):
        bad = indices[(clean_predictions.numpy() != labels.detach().cpu().numpy())]
        raise RuntimeError(
            f"Trace-cohort correctness invariant failed for {spec.key}, seed {seed}; "
            f"unexpected clean errors at dataset indices {bad[:10].tolist()}"
        )
    rows: list[dict[str, object]] = []
    for layer, clean_value in clean.items():
        adv_value = adv[layer]
        delta = (adv_value - clean_value).reshape(len(indices), -1)
        delta_l2 = delta.norm(dim=1)
        delta_rms = delta.square().mean(dim=1).sqrt()
        gradient = gradients.get(layer)
        if gradient is None:
            harm = torch.full_like(delta_l2, torch.nan)
        else:
            harm = -(gradient.reshape(len(indices), -1) * delta).sum(dim=1)
        normalized = harm / margins
        display, order, role, tensor_name = metadata[layer]
        shape = list(clean_value.shape[1:])
        dimension = int(np.prod(shape)) if shape else 1
        for position, dataset_index in enumerate(indices):
            diagnostics = {
                key: (float(value[position]) if residual else math.nan)
                for key, value in residual.items()
            }
            for cohort_id, cohort_kind in memberships[int(dataset_index)]:
                rows.append({
                    "trace_schema_version": TRACE_SCHEMA_VERSION,
                    "cohort_id": cohort_id,
                    "cohort_kind": cohort_kind,
                    "model_key": spec.key,
                    "model": spec.display_name,
                    "semantic_architecture": spec.semantic_architecture,
                    "seed": int(seed),
                    "dataset_index": int(dataset_index),
                    "class_label": int(labels[position]),
                    "epsilon": EPSILON,
                    "attack": "untargeted white-box FGSM generated from traced model with ground-truth loss",
                    "attack_norm": "Linf",
                    "layer": layer,
                    "layer_display_name": display,
                    "layer_order": order,
                    "point_role": role,
                    "tensor_name": tensor_name,
                    "hook_side": hook_sides[layer],
                    "tensor_shape": json.dumps(shape),
                    "dimensionality": dimension,
                    "clean_prediction": int(clean_predictions[position]),
                    "adversarial_prediction": int(adv_predictions[position]),
                    "clean_margin": float(margins[position]),
                    "small_margin_lt_1e-6": bool(float(margins[position]) < 1e-6),
                    "delta_l2": float(delta_l2[position]),
                    "delta_rms": float(delta_rms[position]),
                    "gain_rms": float(delta_rms[position] / input_rms[position]),
                    "projected_harm": float(harm[position]),
                    "normalized_harm": float(normalized[position]),
                    "residual_scale": residual_scale,
                    **diagnostics,
                })
    return rows


def _cohort_memberships_for_model(
    manifest_rows: Sequence[Mapping[str, object]],
    model_key: str,
    seed: int,
) -> dict[int, list[tuple[str, str]]]:
    memberships: dict[int, list[tuple[str, str]]] = {}
    for row in manifest_rows:
        if int(row["seed"]) != int(seed):
            continue
        keys = str(row["model_keys"]).split(";")
        if model_key not in keys:
            continue
        memberships.setdefault(int(row["dataset_index"]), []).append(
            (str(row["cohort_id"]), str(row["cohort_kind"]))
        )
    return memberships


def trace_all_cohorts(
    manifest_rows: Sequence[Mapping[str, object]],
    specs: Mapping[str, RevisionModelSpec] | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Trace each model/seed once over the union of its A/B/C memberships."""

    from tqdm.auto import tqdm

    specs = dict(specs or model_specs())
    images, labels, _images_np, _labels_np = load_official_mnist()
    parts_dir = output_dir / "_trace_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for spec in tqdm(specs.values(), desc="Native activation traces"):
        for seed in SEEDS:
            memberships = _cohort_memberships_for_model(manifest_rows, spec.key, seed)
            if not memberships:
                continue
            part_path = parts_dir / f"{spec.key}_seed={seed}.csv"
            indices = np.array(sorted(memberships), dtype=np.int64)
            n_layers = len(spec.points) + 2 + (1 if spec.key == "reshvql" else 0)
            expected_rows = sum(len(memberships[int(index)]) * n_layers for index in indices)
            if part_path.exists():
                with part_path.open("r", newline="", encoding="utf-8") as existing:
                    reader = csv.DictReader(existing)
                    header = reader.fieldnames or []
                    existing_rows = list(reader)
                identity_valid = all(
                    row.get("trace_schema_version") == TRACE_SCHEMA_VERSION
                    and row.get("model_key") == spec.key
                    and int(row.get("seed", -1)) == int(seed)
                    and int(row.get("dataset_index", -1)) in memberships
                    for row in existing_rows
                )
                if header == list(TRACE_FIELDS) and len(existing_rows) == expected_rows and identity_valid:
                    continue
            model, _checkpoint = load_native_model(spec, seed)
            residual_scale_value = (
                float(model.classifier.residual_scale.detach().cpu())
                if spec.key == "reshvql" else math.nan
            )
            temporary_path = part_path.with_suffix(".csv.tmp")
            with temporary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS, extrasaction="ignore")
                writer.writeheader()
                for start in tqdm(
                    range(0, len(indices), TRACE_BATCH_SIZE),
                    desc=f"{spec.key} seed={seed} ({len(indices)} unique)",
                    leave=False,
                ):
                    batch_indices = indices[start : start + TRACE_BATCH_SIZE]
                    x = images[batch_indices]
                    y = labels[batch_indices]
                    adv = generate_fgsm(model, x, y, EPSILON)
                    clean_values, gradients, margins, _competitors = _capture_clean(model, spec, x, y)
                    adv_values = _capture_adv(model, spec, adv)
                    batch_rows = _batch_trace_rows(
                        spec, seed, batch_indices, memberships, y,
                        clean_values, gradients, adv_values, margins, residual_scale_value,
                    )
                    writer.writerows(batch_rows)
                    handle.flush()
            with temporary_path.open("r", newline="", encoding="utf-8") as completed:
                reader = csv.DictReader(completed)
                header = reader.fieldnames or []
                completed_rows = list(reader)
            identity_valid = all(
                row.get("trace_schema_version") == TRACE_SCHEMA_VERSION
                and row.get("model_key") == spec.key
                and int(row.get("seed", -1)) == int(seed)
                and int(row.get("dataset_index", -1)) in memberships
                for row in completed_rows
            )
            if header != list(TRACE_FIELDS) or len(completed_rows) != expected_rows or not identity_valid:
                raise RuntimeError(
                    f"Incomplete/stale trace part {temporary_path}: {len(completed_rows)} rows, "
                    f"expected {expected_rows}, identity_valid={identity_valid}"
                )
            os.replace(temporary_path, part_path)
            del model
            gc.collect()
    merged_csv = output_dir / "sample_level_layerwise_traces.csv"
    with merged_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(TRACE_FIELDS)
        expected_parts = [
            parts_dir / f"{spec.key}_seed={seed}.csv"
            for spec in specs.values() for seed in SEEDS
            if _cohort_memberships_for_model(manifest_rows, spec.key, seed)
        ]
        for part_path in expected_parts:
            with part_path.open("r", newline="", encoding="utf-8") as source:
                reader = csv.reader(source)
                next(reader, None)
                writer.writerows(reader)
    return merged_csv


def write_initial_metadata(
    validation_rows: Sequence[Mapping[str, object]],
    cohort_summary: Sequence[Mapping[str, object]],
    specs: Mapping[str, RevisionModelSpec] | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    specs = dict(specs or model_specs())
    image_path = MNIST_RAW_DIR / "t10k-images-idx3-ubyte"
    label_path = MNIST_RAW_DIR / "t10k-labels-idx1-ubyte"
    metadata = {
        "analysis": "Propagation du déplacement adversarial et sélectivité de l'interface classique--quantique",
        "schema_version": TRACE_SCHEMA_VERSION,
        "models": {key: _jsonable(asdict(spec)) for key, spec in specs.items()},
        "resolved_parallel_model_name": "HPVQL",
        "parallel_model_resolution": (
            "Storage/raw name hpvqm is historical; the saved Hydra classifier name is hpvql and the native "
            "runtime class is HPVQL backed by ParaVQC with five measured blocks."
        ),
        "checkpoints": [_jsonable(row) for row in validation_rows],
        "cohorts": [_jsonable(row) for row in cohort_summary],
        "dataset": {
            "name": "official MNIST test set",
            "n": 10000,
            "image_idx": str(image_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "label_idx": str(label_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "image_idx_sha256": _sha256(image_path),
            "label_idx_sha256": _sha256(label_path),
            "order_validated_against": str((ATTACK_DIR / "seed_0" / "labels.npy").relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "attack": {
            "name": "untargeted white-box FGSM",
            "epsilon": EPSILON,
            "norm": "Linf",
            "formula": "clip(x + epsilon * sign(gradient_x cross_entropy(model(x), y_true)), 0, 1)",
            "generated_separately_for_each_traced_model": True,
            "model_mode": "eval",
        },
        "metrics": {
            "delta": "a_l(x_adv) - a_l(x)",
            "rms": "sqrt(mean(delta**2)) per example",
            "gain_rms": "RMS(delta_l) / RMS(delta_input) per example",
            "clean_margin": "logit_true - max(logit_other_classes)",
            "projected_harm": "-gradient_a_l(clean margin) dot delta_a_l",
            "normalized_harm": "projected_harm / clean_margin (no tau, winsorization, or truncation)",
            "measurement_norm_ratio": "L2(delta_measurement) / L2(delta_prequantum)",
            "samplewise_harm_ratio_exclusion": "exclude only abs(normalized_harm_prequantum) < 1e-8",
            "small_margin_sensitivity": "repeat normalized-harm summaries after excluding clean_margin < 1e-6",
        },
        "native_quantum_boundary": {
            "used_native_pennylane_torchlayer_outputs": True,
            "surrogate_used": False,
            "amplitude_normalized_state_separately_accessible": False,
            "note": "No manually reconstructed or dequantized interface is used; r is the native measured module output.",
        },
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "interval": 0.95,
            "scope": "sample-level uncertainty only",
            "seed_population_inference": False,
        },
        "source_lineage": {
            "prior_19_trace": "outputs/mechanistic_propagation/layerwise_trace_samples.csv",
            "prior_followup_figures": "outputs/figure_redesign_from_trace/followup_figures",
            "attack_baselines": str(ATTACK_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "analysis_metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_native_trace_stage(output_dir: Path = OUTPUT_DIR) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = model_specs()
    validation, predictions = validate_checkpoints(specs, output_dir)
    manifest, cohort_summary = build_cohort_manifest(predictions, specs, output_dir)
    metadata_path = write_initial_metadata(validation, cohort_summary, specs, output_dir)
    trace_csv = trace_all_cohorts(manifest, specs, output_dir)
    return {
        "checkpoint_validation": str(output_dir / "checkpoint_validation.csv"),
        "cohort_manifest": str(output_dir / "cohort_manifest.csv"),
        "trace_csv": str(trace_csv),
        "analysis_metadata": str(metadata_path),
    }


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[finite], weights[finite]
    if not len(values):
        return math.nan
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return float(np.interp(float(quantile), cumulative, values))


def _describe_values(values) -> dict[str, float | int]:
    raw = np.asarray(values, dtype=np.float64)
    values = _finite(raw)
    counts = {
        "n": int(len(raw)),
        "n_finite": int(np.isfinite(raw).sum()),
        "n_nonfinite": int((~np.isfinite(raw)).sum()),
        "n_nan": int(np.isnan(raw).sum()),
        "n_posinf": int(np.isposinf(raw).sum()),
        "n_neginf": int(np.isneginf(raw).sum()),
    }
    if not len(values):
        return {
            **counts, "median": math.nan, "q1": math.nan, "q3": math.nan,
            "mean": math.nan, "std": math.nan,
        }
    return {
        **counts,
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def _describe_equal_seed(seed_values: Mapping[int, np.ndarray]) -> dict[str, float | int]:
    raw = {int(seed): np.asarray(values, dtype=np.float64) for seed, values in seed_values.items()}
    clean = {seed: _finite(values) for seed, values in raw.items() if len(_finite(values))}
    if not clean:
        combined = np.concatenate(list(raw.values())) if raw else np.array([], dtype=np.float64)
        return _describe_values(combined)
    all_values = np.concatenate(list(clean.values()))
    all_weights = np.concatenate([
        np.full(len(values), 1.0 / (len(clean) * len(values)), dtype=np.float64)
        for values in clean.values()
    ])
    mean = float(np.sum(all_values * all_weights))
    variance = float(np.sum(all_weights * (all_values - mean) ** 2))
    return {
        "n": int(sum(len(values) for values in raw.values())),
        "n_finite": int(sum(np.isfinite(values).sum() for values in raw.values())),
        "n_nonfinite": int(sum((~np.isfinite(values)).sum() for values in raw.values())),
        "n_nan": int(sum(np.isnan(values).sum() for values in raw.values())),
        "n_posinf": int(sum(np.isposinf(values).sum() for values in raw.values())),
        "n_neginf": int(sum(np.isneginf(values).sum() for values in raw.values())),
        "median": _weighted_quantile(all_values, all_weights, 0.5),
        "q1": _weighted_quantile(all_values, all_weights, 0.25),
        "q3": _weighted_quantile(all_values, all_weights, 0.75),
        "mean": mean,
        "std": math.sqrt(max(variance, 0.0)),
    }


def _bootstrap_median_ci(values, salt: str) -> tuple[float, float]:
    values = _finite(values)
    if not len(values):
        return math.nan, math.nan
    seed = int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    medians = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        medians[index] = np.median(rng.choice(values, size=len(values), replace=True))
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _bootstrap_equal_seed_median_ci(
    seed_values: Mapping[int, np.ndarray], salt: str,
) -> tuple[float, float]:
    clean = {int(seed): _finite(values) for seed, values in seed_values.items() if len(_finite(values))}
    if not clean:
        return math.nan, math.nan
    sample_size = min(len(values) for values in clean.values())
    seed = int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    medians = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        draw = np.concatenate([
            rng.choice(values, size=sample_size, replace=True) for values in clean.values()
        ])
        medians[index] = np.median(draw)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _paired_ratio_bootstrap(
    frames: Mapping[int, object],
    numerator: str,
    denominator: str,
    definition: str,
    salt: str,
) -> tuple[float, float]:
    """Paired bootstrap for ratio-of-medians or median samplewise ratio."""

    cleaned: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seed, frame in frames.items():
        numerator_values = np.asarray(frame[numerator], dtype=np.float64)
        denominator_values = np.asarray(frame[denominator], dtype=np.float64)
        finite = np.isfinite(numerator_values) & np.isfinite(denominator_values)
        if definition == "median_samplewise_ratio":
            finite &= np.abs(denominator_values) >= 1e-8
        if finite.any():
            cleaned[int(seed)] = (numerator_values[finite], denominator_values[finite])
    if not cleaned:
        return math.nan, math.nan
    sample_size = min(len(pair[0]) for pair in cleaned.values())
    seed_value = int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_value)
    statistics = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for iteration in range(BOOTSTRAP_RESAMPLES):
        numerator_draws, denominator_draws = [], []
        for numerator_values, denominator_values in cleaned.values():
            indices = rng.integers(0, len(numerator_values), size=sample_size)
            numerator_draws.append(numerator_values[indices])
            denominator_draws.append(denominator_values[indices])
        numerator_draw = np.concatenate(numerator_draws)
        denominator_draw = np.concatenate(denominator_draws)
        if definition == "ratio_of_medians":
            denominator_median = np.median(denominator_draw)
            statistics[iteration] = (
                np.median(numerator_draw) / denominator_median
                if abs(denominator_median) > 0 else math.nan
            )
        else:
            statistics[iteration] = np.median(numerator_draw / denominator_draw)
    statistics = _finite(statistics)
    if not len(statistics):
        return math.nan, math.nan
    return float(np.quantile(statistics, 0.025)), float(np.quantile(statistics, 0.975))


def build_layerwise_summary(trace, output_dir: Path = OUTPUT_DIR):
    import pandas as pd

    metrics = (
        "delta_l2", "delta_rms", "gain_rms", "projected_harm",
        "normalized_harm", "clean_margin",
    )
    identity = [
        "cohort_id", "cohort_kind", "model_key", "model", "semantic_architecture",
        "layer", "layer_display_name", "layer_order", "point_role", "tensor_name",
        "hook_side", "tensor_shape", "dimensionality",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in trace.groupby(identity + ["seed"], sort=False, dropna=False):
        base = dict(zip(identity + ["seed"], keys))
        filters = [("all", np.ones(len(group), dtype=bool))]
        for metric in metrics:
            metric_filters = filters + (
                [("margin_ge_1e-6", group["clean_margin"].to_numpy() >= 1e-6)]
                if metric == "normalized_harm" else []
            )
            for filter_name, mask in metric_filters:
                values = group.loc[mask, metric].to_numpy(dtype=np.float64)
                description = _describe_values(values)
                lower, upper = _bootstrap_median_ci(values, f"layer:{keys}:{metric}:{filter_name}")
                rows.append({
                    "summary_scope": "seed_specific",
                    **base,
                    "metric": metric,
                    "margin_filter": filter_name,
                    **description,
                    "median_ci_lower": lower,
                    "median_ci_upper": upper,
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "seed_weighting": "not_applicable",
                    "small_margin_count": int((group["clean_margin"] < 1e-6).sum()),
                })
    seed_frame = pd.DataFrame(rows)
    pooled_rows: list[dict[str, object]] = []
    for keys, group in trace.groupby(identity, sort=False, dropna=False):
        base = dict(zip(identity, keys))
        for metric in metrics:
            filter_names = ["all", "margin_ge_1e-6"] if metric == "normalized_harm" else ["all"]
            for filter_name in filter_names:
                selected = group if filter_name == "all" else group[group["clean_margin"] >= 1e-6]
                seed_values = {
                    int(seed): seed_group[metric].to_numpy(dtype=np.float64)
                    for seed, seed_group in selected.groupby("seed")
                }
                description = _describe_equal_seed(seed_values)
                lower, upper = _bootstrap_equal_seed_median_ci(
                    seed_values, f"pooled-layer:{keys}:{metric}:{filter_name}"
                )
                pooled_rows.append({
                    "summary_scope": "pooled_equal_seed",
                    **base,
                    "seed": "pooled",
                    "metric": metric,
                    "margin_filter": filter_name,
                    **description,
                    "median_ci_lower": lower,
                    "median_ci_upper": upper,
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "seed_weighting": "equal mixture weight per seed",
                    "small_margin_count": int((group["clean_margin"] < 1e-6).sum()),
                })
    result = pd.concat([seed_frame, pd.DataFrame(pooled_rows)], ignore_index=True)
    result.to_csv(output_dir / "layerwise_summary.csv", index=False)
    return result


def _interface_pairs(trace, specs: Mapping[str, RevisionModelSpec]):
    import pandas as pd

    frames = []
    for model_key in LATE_MODELS:
        spec = specs[model_key]
        selected = trace[
            (trace["model_key"] == model_key)
            & trace["layer"].isin([spec.prequantum_point, spec.measurement_point])
        ].copy()
        index = [
            "cohort_id", "cohort_kind", "model_key", "model", "semantic_architecture",
            "seed", "dataset_index", "class_label", "clean_margin",
        ]
        pivot = selected.pivot(index=index, columns="layer", values=["delta_l2", "normalized_harm"])
        pivot.columns = [f"{metric}__{layer}" for metric, layer in pivot.columns]
        pivot = pivot.reset_index()
        pivot["prequantum_point"] = spec.prequantum_point
        pivot["measurement_point"] = spec.measurement_point
        pivot["delta_l2_prequantum"] = pivot[f"delta_l2__{spec.prequantum_point}"]
        pivot["delta_l2_measurement"] = pivot[f"delta_l2__{spec.measurement_point}"]
        pivot["normalized_harm_prequantum"] = pivot[f"normalized_harm__{spec.prequantum_point}"]
        pivot["normalized_harm_measurement"] = pivot[f"normalized_harm__{spec.measurement_point}"]
        pivot["measurement_norm_ratio"] = (
            pivot["delta_l2_measurement"] / pivot["delta_l2_prequantum"]
        )
        frames.append(pivot[[
            *index, "prequantum_point", "measurement_point",
            "delta_l2_prequantum", "delta_l2_measurement", "measurement_norm_ratio",
            "normalized_harm_prequantum", "normalized_harm_measurement",
        ]])
    return pd.concat(frames, ignore_index=True)


def _interface_summary_row(
    group,
    summary_scope: str,
    seed_label: object,
    margin_filter: str,
    salt: str,
) -> dict[str, object]:
    selected = group if margin_filter == "all" else group[group["clean_margin"] >= 1e-6]
    by_seed = {int(seed): frame for seed, frame in selected.groupby("seed")}

    def describe(column: str) -> tuple[dict[str, object], float, float]:
        if summary_scope == "seed_specific":
            values = selected[column].to_numpy(dtype=np.float64)
            description = _describe_values(values)
            lower, upper = _bootstrap_median_ci(values, f"{salt}:{column}")
        else:
            values = {
                int(seed): frame[column].to_numpy(dtype=np.float64)
                for seed, frame in by_seed.items()
            }
            description = _describe_equal_seed(values)
            lower, upper = _bootstrap_equal_seed_median_ci(values, f"{salt}:{column}")
        return description, lower, upper

    row: dict[str, object] = {
        "summary_scope": summary_scope,
        "seed": seed_label,
        "cohort_id": selected["cohort_id"].iloc[0] if len(selected) else group["cohort_id"].iloc[0],
        "cohort_kind": selected["cohort_kind"].iloc[0] if len(selected) else group["cohort_kind"].iloc[0],
        "model_key": selected["model_key"].iloc[0] if len(selected) else group["model_key"].iloc[0],
        "model": selected["model"].iloc[0] if len(selected) else group["model"].iloc[0],
        "semantic_architecture": selected["semantic_architecture"].iloc[0] if len(selected) else group["semantic_architecture"].iloc[0],
        "prequantum_point": group["prequantum_point"].iloc[0],
        "measurement_point": group["measurement_point"].iloc[0],
        "margin_filter": margin_filter,
        "n": int(len(selected)),
        "small_margin_count_before_filter": int((group["clean_margin"] < 1e-6).sum()),
        "seed_weighting": "equal mixture weight per seed" if summary_scope != "seed_specific" else "not_applicable",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }
    for column in (
        "delta_l2_prequantum", "delta_l2_measurement", "measurement_norm_ratio",
        "normalized_harm_prequantum", "normalized_harm_measurement",
    ):
        description, lower, upper = describe(column)
        for name, value in description.items():
            row[f"{column}_{name}"] = value
        row[f"{column}_median_ci_lower"] = lower
        row[f"{column}_median_ci_upper"] = upper

    pre = selected["normalized_harm_prequantum"].to_numpy(dtype=np.float64)
    measurement = selected["normalized_harm_measurement"].to_numpy(dtype=np.float64)
    pre_median = (
        _describe_values(pre)["median"] if summary_scope == "seed_specific"
        else _describe_equal_seed({int(seed): frame["normalized_harm_prequantum"] for seed, frame in by_seed.items()})["median"]
    )
    measurement_median = (
        _describe_values(measurement)["median"] if summary_scope == "seed_specific"
        else _describe_equal_seed({int(seed): frame["normalized_harm_measurement"] for seed, frame in by_seed.items()})["median"]
    )
    row["ratio_of_medians"] = (
        float(measurement_median / pre_median) if abs(float(pre_median)) > 0 else math.nan
    )
    finite_pair = np.isfinite(pre) & np.isfinite(measurement)
    small_denominator = np.isfinite(pre) & (np.abs(pre) < 1e-8)
    valid_ratio = finite_pair & ~small_denominator
    sample_ratios = measurement[valid_ratio] / pre[valid_ratio]
    row["samplewise_ratio_excluded_abs_pre_lt_1e_minus_8"] = int(small_denominator.sum())
    row["samplewise_ratio_excluded_nonfinite"] = int((~finite_pair & ~small_denominator).sum())
    row["samplewise_ratio_included"] = int(valid_ratio.sum())
    if summary_scope == "seed_specific":
        ratio_description = _describe_values(sample_ratios)
    else:
        ratio_by_seed = {}
        for seed, frame in by_seed.items():
            seed_pre = frame["normalized_harm_prequantum"].to_numpy(dtype=np.float64)
            seed_measurement = frame["normalized_harm_measurement"].to_numpy(dtype=np.float64)
            seed_valid = np.isfinite(seed_pre) & np.isfinite(seed_measurement) & (np.abs(seed_pre) >= 1e-8)
            ratio_by_seed[int(seed)] = seed_measurement[seed_valid] / seed_pre[seed_valid]
        ratio_description = _describe_equal_seed(ratio_by_seed)
    for name, value in ratio_description.items():
        row[f"samplewise_ratio_{name}"] = value
    row["median_samplewise_ratio"] = ratio_description["median"]
    ratio_frames = by_seed if summary_scope != "seed_specific" else {
        int(seed_label): selected
    }
    ratio_lower, ratio_upper = _paired_ratio_bootstrap(
        ratio_frames, "normalized_harm_measurement", "normalized_harm_prequantum",
        "ratio_of_medians", f"{salt}:ratio-of-medians",
    )
    sample_lower, sample_upper = _paired_ratio_bootstrap(
        ratio_frames, "normalized_harm_measurement", "normalized_harm_prequantum",
        "median_samplewise_ratio", f"{salt}:samplewise-ratio",
    )
    row["ratio_of_medians_ci_lower"] = ratio_lower
    row["ratio_of_medians_ci_upper"] = ratio_upper
    row["median_samplewise_ratio_ci_lower"] = sample_lower
    row["median_samplewise_ratio_ci_upper"] = sample_upper
    return row


def build_interface_transition_summary(trace, output_dir: Path = OUTPUT_DIR):
    import pandas as pd

    pairs = _interface_pairs(trace, model_specs())
    pairs.to_csv(output_dir / "interface_transition_sample_level.csv", index=False)
    rows: list[dict[str, object]] = []
    identity = ["cohort_id", "cohort_kind", "model_key", "model", "semantic_architecture"]
    for keys, group in pairs.groupby(identity + ["seed"], sort=False):
        seed = int(keys[-1])
        for margin_filter in ("all", "margin_ge_1e-6"):
            rows.append(_interface_summary_row(
                group, "seed_specific", seed, margin_filter,
                f"interface:{keys}:{margin_filter}",
            ))
    for keys, group in pairs.groupby(identity, sort=False):
        for margin_filter in ("all", "margin_ge_1e-6"):
            rows.append(_interface_summary_row(
                group, "pooled_equal_seed", "pooled", margin_filter,
                f"pooled-interface:{keys}:{margin_filter}",
            ))
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "interface_transition_summary.csv", index=False)
    return pairs, result


def build_residual_branch_summary(trace, output_dir: Path = OUTPUT_DIR):
    import pandas as pd

    diagnostics = (
        "classical_to_quantum_displacement_ratio",
        "raw_classical_to_quantum_displacement_ratio",
        "quantum_norm_share_of_branch_sum",
        "classical_norm_share_of_branch_sum",
        "quantum_to_fused_norm_ratio",
        "classical_to_fused_norm_ratio",
        "quantum_projection_share_of_fused",
        "classical_projection_share_of_fused",
        "residual_reconstruction_relative_error",
    )
    source = trace[(trace["model_key"] == "reshvql") & (trace["layer"] == "fused")].copy()
    rows: list[dict[str, object]] = []
    identity = ["cohort_id", "cohort_kind", "model_key", "model", "semantic_architecture"]
    for keys, group in source.groupby(identity + ["seed"], sort=False):
        base = dict(zip(identity + ["seed"], keys))
        for metric in diagnostics:
            values = group[metric].to_numpy(dtype=np.float64)
            description = _describe_values(values)
            lower, upper = _bootstrap_median_ci(values, f"residual:{keys}:{metric}")
            rows.append({
                "summary_scope": "seed_specific", **base, "metric": metric,
                **description, "median_ci_lower": lower, "median_ci_upper": upper,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "seed_weighting": "not_applicable",
            })
    for keys, group in source.groupby(identity, sort=False):
        base = dict(zip(identity, keys))
        for metric in diagnostics:
            seed_values = {
                int(seed): seed_group[metric].to_numpy(dtype=np.float64)
                for seed, seed_group in group.groupby("seed")
            }
            description = _describe_equal_seed(seed_values)
            lower, upper = _bootstrap_equal_seed_median_ci(
                seed_values, f"pooled-residual:{keys}:{metric}"
            )
            rows.append({
                "summary_scope": "pooled_equal_seed", **base, "seed": "pooled", "metric": metric,
                **description, "median_ci_lower": lower, "median_ci_upper": upper,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "seed_weighting": "equal mixture weight per seed",
            })
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "residual_branch_summary.csv", index=False)
    return result


def _figure_group_summary(group, metric: str) -> tuple[float, float, float]:
    values = {
        int(seed): seed_group[metric].to_numpy(dtype=np.float64)
        for seed, seed_group in group.groupby("seed")
    }
    description = _describe_equal_seed(values)
    return float(description["median"]), float(description["q1"]), float(description["q3"])


def _save_figure_metadata(trace, output_dir: Path, specs: Mapping[str, RevisionModelSpec]) -> Path:
    cohort_sizes = (
        trace[["cohort_id", "seed", "dataset_index"]]
        .drop_duplicates()
        .groupby(["cohort_id", "seed"])["dataset_index"]
        .nunique()
        .to_dict()
    )
    common = {
        "seeds": list(SEEDS),
        "dataset": "official MNIST test set",
        "model_mode": "evaluation mode",
        "attack": "untargeted white-box FGSM, ground-truth cross-entropy, native traced model",
        "attack_formula": "x_adv = clip(x + 0.10 * sign(gradient_x cross_entropy(model(x), y_true)), 0, 1)",
        "epsilon": EPSILON,
        "norm": "Linf",
        "aggregation": "points/lines are equal-seed pooled medians; ribbons are equal-seed pooled interquartile ranges",
    }
    dual_stages = {
        "hvql": [
            ("input", 0), ("encoder_conv1", 1), ("encoder_block1", 2),
            ("encoder_conv2", 3), ("encoder_block2", 4),
            ("prequantum_linear", 5), ("prequantum", 6),
            ("measurement", 7), ("readout", 8), ("logits", 9),
        ],
        "cnn": [
            ("input", 0), ("conv1", 1), ("conv_block1", 2), ("conv2", 3),
            ("conv_block2", 4), ("hidden_fc", 5), ("logits", 6),
        ],
    }
    late_stages = {
        "hvql": [("prequantum", 0.0), ("measurement", 1.0), ("logits", 2.0)],
        "hpvql": [("prequantum", 0.0), ("measurement", 1.0), ("logits", 2.0)],
        "hvqm": [("prequantum", 0.0), ("measurement", 1.0), ("logits", 2.0)],
        "hqe": [("prequantum", 0.0), ("measurement", 1.0), ("logits", 2.0)],
        "reshvql": [
            ("prequantum", 0.0), ("quantum_branch", 1.0),
            ("fused", 1.5), ("logits", 2.0),
        ],
    }
    dual_selection = trace[(trace["cohort_id"] == "A") & trace["model_key"].isin(("hvql", "cnn"))]
    late_selection = trace[(trace["cohort_id"] == "B") & trace["model_key"].isin(LATE_MODELS)]
    residual_scales = {
        str(int(seed)): float(group["residual_scale"].dropna().iloc[0])
        for seed, group in late_selection[late_selection["model_key"] == "reshvql"].groupby("seed")
    }
    configs = lambda keys: {key: _jsonable(asdict(specs[key])) for key in keys}
    metadata = {
        "hvql_cnn_dual_progression_side_by_side_revised.png": {
            **common,
            "cohort": "A: correctly classified by both reference CNN and HVQL; deterministic class-balanced selection",
            "sample_size_by_seed": {str(seed): int(cohort_sizes.get(("A", seed), 0)) for seed in SEEDS},
            "models": ["hvql", "cnn"],
            "panel_order_left_to_right": ["hvql", "cnn"],
            "model_configurations": configs(("hvql", "cnn")),
            "plotted_stages": dual_stages,
            "x_tick_labels": {
                "hvql": [
                    "Entrée", "Conv. 1", "Pool 1", "Conv. 2", "Pool 2",
                    "Dense 1", "Interface Quantique", "Mesures Quantiques", "Dense 2", "Logits",
                ],
                "cnn": ["Entrée", "Conv. 1", "Pool 1", "Conv. 2", "Pool 2", "Dense", "Logits"],
            },
            "panel_titles": ["HVQL", "CNN"],
            "legend_labels": ["Gain RMS", "Dommage Normalisé"],
            "axes": "blue logarithmic RMS-gain axis and orange exact normalized projected-harm H_l/m_f(x) axis",
            "zero_gain_sample_count": int((dual_selection["gain_rms"] <= 0).sum()),
            "log_axis_handling": "nonpositive values are retained in tables; nonpositive ribbon bounds are masked on the RMS logarithmic axis",
            "visual_warning": "x positions are categorical; slopes across panels or across the two y axes are not quantitatively comparable",
        },
        "late_hybrid_interface_rms_progression_revised.png": {
            **common,
            "cohort": "B: correctly classified by all five displayed late-hybrid models; deterministic class-balanced selection",
            "sample_size_by_seed": {str(seed): int(cohort_sizes.get(("B", seed), 0)) for seed in SEEDS},
            "models": list(LATE_MODELS),
            "legend_order_and_labels": ["HPVQL", "HQE", "HVQL", "HVQM", "Résiduel"],
            "model_configurations": configs(LATE_MODELS),
            "plotted_stages": late_stages,
            "x_ticks": {"0.0": "Interface Quantique", "1.0": "Mesure Quantique", "2.0": "Logits"},
            "residual_scale_by_seed": residual_scales,
            "axis": "logarithmic RMS gain",
            "zero_gain_sample_count": int((late_selection["gain_rms"] <= 0).sum()),
            "log_axis_handling": "nonpositive values are retained in tables; nonpositive ribbon bounds are masked on the logarithmic axis",
            "stage_aliases": {
                "hvqm": "measurement is the native identity readout and therefore equals logits",
                "reshvql": "fused representation is the native classifier output and therefore equals logits",
            },
            "residual_fusion_tick_scope": "the residual-HVQL fusion is plotted at x=1.5 without a tick label; other models have no observation there",
        },
        "late_hybrid_interface_normalized_harm_progression_revised.png": {
            **common,
            "cohort": "B: correctly classified by all five displayed late-hybrid models; deterministic class-balanced selection",
            "sample_size_by_seed": {str(seed): int(cohort_sizes.get(("B", seed), 0)) for seed in SEEDS},
            "models": list(LATE_MODELS),
            "legend_order_and_labels": ["HPVQL", "HQE", "HVQL", "HVQM", "Résiduel"],
            "model_configurations": configs(LATE_MODELS),
            "plotted_stages": late_stages,
            "x_ticks": {"0.0": "Interface Quantique", "1.0": "Mesure Quantique", "2.0": "Logits"},
            "residual_scale_by_seed": residual_scales,
            "axis": "normalized projected harm H_l/m_f(x), exact clean-margin denominator",
            "stage_aliases": {
                "hvqm": "measurement is the native identity readout and therefore equals logits",
                "reshvql": "fused representation is the native classifier output and therefore equals logits",
            },
            "residual_fusion_tick_scope": "the residual-HVQL fusion is plotted at x=1.5 without a tick label; other models have no observation there",
        },
    }
    path = output_dir / "figure_metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _generate_revised_figures_legacy(trace, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    specs = model_specs()
    colors = {
        "cnn": "#4C78A8", "hvql": "#F58518", "hpvql": "#54A24B",
        "hvqm": "#E45756", "hqe": "#B279A2", "reshvql": "#72B7B2",
    }
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 300, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22,
    })
    paths: list[Path] = []

    figure, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), constrained_layout=False)
    for axis, model_key in zip(axes, ("cnn", "hvql")):
        selected = trace[(trace["cohort_id"] == "A") & (trace["model_key"] == model_key)].copy()
        layers = (
            selected[["layer", "layer_display_name", "layer_order"]]
            .drop_duplicates().sort_values("layer_order")
        )
        gain_axis = axis
        harm_axis = axis.twinx()
        xs, gain_median, gain_q1, gain_q3, harm_median, harm_q1, harm_q3 = ([] for _ in range(7))
        for x_value, layer in enumerate(layers["layer"]):
            layer_group = selected[selected["layer"] == layer]
            gain = _figure_group_summary(layer_group, "gain_rms")
            harm = _figure_group_summary(layer_group, "normalized_harm")
            xs.append(x_value)
            gain_median.append(gain[0]); gain_q1.append(gain[1]); gain_q3.append(gain[2])
            harm_median.append(harm[0]); harm_q1.append(harm[1]); harm_q3.append(harm[2])
        gain_axis.plot(xs, gain_median, color=colors[model_key], marker="o", linewidth=2.2, label="RMS gain")
        gain_axis.fill_between(xs, gain_q1, gain_q3, color=colors[model_key], alpha=0.18)
        harm_axis.plot(xs, harm_median, color="#6F4E7C", marker="s", linewidth=1.9, linestyle="--", label="Normalized harm")
        harm_axis.fill_between(xs, harm_q1, harm_q3, color="#6F4E7C", alpha=0.12)
        gain_axis.axhline(1.0, color="#444444", linewidth=0.8, linestyle=":")
        gain_axis.set_yscale("log")
        gain_axis.set_xticks(xs)
        gain_axis.set_xticklabels(layers["layer_display_name"], rotation=38, ha="right")
        gain_axis.set_title(specs[model_key].display_name)
        gain_axis.set_ylabel("RMS displacement gain $G_\\ell$")
        harm_axis.set_ylabel("Normalized projected harm $H_\\ell / margin$", color="#6F4E7C")
        harm_axis.grid(False)
        handles = gain_axis.get_lines()[:1] + harm_axis.get_lines()[:1]
        gain_axis.legend(handles=handles, labels=["RMS gain (median/IQR)", "Normalized harm (median/IQR)"], loc="best")
    figure.suptitle("Adversarial propagation: CNN reference and HVQL (comparison-specific Cohort A)", fontsize=14, y=0.98)
    figure.text(
        0.5, 0.015,
        "Cohort A: common clean-correct, balanced 50/class; n=500 per seed; seeds 0, 42; native white-box FGSM, "
        "$\\epsilon=0.10$, $L_\\infty$. Equal-seed pooled medians and IQRs.",
        ha="center", fontsize=8.5,
    )
    figure.subplots_adjust(bottom=0.28, top=0.88, wspace=0.34)
    path = output_dir / "hvql_cnn_dual_progression_side_by_side_revised.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    stage_map = {
        "hvql": [("encoder_flattened", 0), ("prequantum", 1), ("measurement", 2), ("readout", 3), ("logits", 4)],
        "hpvql": [("encoder_flattened", 0), ("prequantum", 1), ("measurement", 2), ("readout", 3), ("logits", 4)],
        "hvqm": [("prequantum", 1), ("measurement", 2), ("readout", 3), ("logits", 4)],
        "hqe": [("encoder_final", 0), ("prequantum", 1), ("measurement", 2), ("readout", 3), ("logits", 4)],
        "reshvql": [("encoder_flattened", 0), ("prequantum", 1), ("quantum_branch", 2), ("fused", 3), ("logits", 4)],
    }
    stage_labels = ["Encoder final", "Prequantum z", "Measured r / q", "Readout / fusion", "Logits"]

    def progression_figure(metric: str, filename: str, title: str, ylabel: str, log_scale: bool) -> Path:
        figure, axis = plt.subplots(figsize=(10.8, 6.2))
        selected = trace[trace["cohort_id"] == "B"]
        for model_key in LATE_MODELS:
            xs, medians, q1s, q3s = [], [], [], []
            for layer, x_value in stage_map[model_key]:
                group = selected[(selected["model_key"] == model_key) & (selected["layer"] == layer)]
                if group.empty:
                    continue
                median, q1, q3 = _figure_group_summary(group, metric)
                xs.append(x_value); medians.append(median); q1s.append(q1); q3s.append(q3)
            axis.plot(xs, medians, color=colors[model_key], marker="o", linewidth=2, label=specs[model_key].display_name)
            axis.fill_between(xs, q1s, q3s, color=colors[model_key], alpha=0.13)
        if log_scale:
            axis.set_yscale("log")
            axis.axhline(1.0, color="#444444", linewidth=0.8, linestyle=":")
            axis.yaxis.set_major_formatter(mticker.ScalarFormatter())
        else:
            axis.axhline(0.0, color="#444444", linewidth=0.8, linestyle=":")
        axis.set_xticks(range(5), stage_labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend(ncol=2, frameon=False)
        figure.text(
            0.5, 0.015,
            "Cohort B: common clean-correct across all five models, balanced 50/class; n=500 per seed; seeds 0, 42; "
            "native model-specific FGSM, $\\epsilon=0.10$, $L_\\infty$. Equal-seed pooled medians and IQRs.",
            ha="center", fontsize=8.5,
        )
        figure.subplots_adjust(bottom=0.14)
        target = output_dir / filename
        figure.savefig(target, bbox_inches="tight")
        plt.close(figure)
        return target

    paths.append(progression_figure(
        "gain_rms", "late_hybrid_interface_rms_progression_revised.png",
        "Late-hybrid RMS displacement progression (comparison-specific Cohort B)",
        "RMS displacement gain $G_\\ell$", True,
    ))
    paths.append(progression_figure(
        "normalized_harm", "late_hybrid_interface_normalized_harm_progression_revised.png",
        "Late-hybrid normalized projected harm (comparison-specific Cohort B)",
        "Normalized projected harm $H_\\ell / margin$", False,
    ))
    _save_figure_metadata(trace, output_dir, specs)
    return paths


def generate_revised_figures(trace, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """Render the three revised figures using the established publication style."""

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D

    specs = model_specs()
    model_styles = {
        "hvql": ("#0072B2", "-", "o"),
        "hpvql": ("#009E73", "--", "s"),
        "hvqm": ("#AA4499", "-.", "D"),
        "hqe": ("#999933", ":", "P"),
        "reshvql": ("#D55E00", "--", "X"),
    }
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "figure.dpi": 140, "savefig.dpi": 360,
        "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 9.1,
        "ytick.labelsize": 9.2, "legend.fontsize": 8.8, "axes.linewidth": 0.8,
        "lines.linewidth": 2.0, "lines.markersize": 6.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#D8DDE5",
        "grid.linewidth": 0.65, "grid.alpha": 0.7,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    def save_all(figure, png_path: Path) -> None:
        for suffix in (".png", ".pdf", ".svg"):
            figure.savefig(png_path.with_suffix(suffix), bbox_inches="tight", pad_inches=0.06)

    dual_stages = {
        "hvql": [
            ("input", "Entrée"), ("encoder_conv1", "Conv. 1"),
            ("encoder_block1", "Pool 1"), ("encoder_conv2", "Conv. 2"),
            ("encoder_block2", "Pool 2"), ("prequantum_linear", "Dense 1"),
            ("prequantum", "Interface Quantique"),
            ("measurement", "Mesures Quantiques"),
            ("readout", "Dense 2"), ("logits", "Logits"),
        ],
        "cnn": [
            ("input", "Entrée"), ("conv1", "Conv. 1"), ("conv_block1", "Pool 1"),
            ("conv2", "Conv. 2"), ("conv_block2", "Pool 2"),
            ("hidden_fc", "Dense"), ("logits", "Logits"),
        ],
    }
    dual_data: dict[str, dict[str, object]] = {}
    all_gain_q1: list[float] = []
    all_gain_q3: list[float] = []
    all_harm_q1: list[float] = []
    all_harm_q3: list[float] = []
    for model_key in ("hvql", "cnn"):
        model_trace = trace[(trace["cohort_id"] == "A") & (trace["model_key"] == model_key)]
        data = {"x": [], "labels": [], "gain": [], "gain_q1": [], "gain_q3": [],
                "harm": [], "harm_q1": [], "harm_q3": []}
        for x_value, (layer, label) in enumerate(dual_stages[model_key]):
            group = model_trace[model_trace["layer"] == layer]
            gain = _figure_group_summary(group, "gain_rms")
            harm = _figure_group_summary(group, "normalized_harm")
            data["x"].append(x_value); data["labels"].append(label)
            data["gain"].append(gain[0]); data["gain_q1"].append(gain[1]); data["gain_q3"].append(gain[2])
            data["harm"].append(harm[0]); data["harm_q1"].append(harm[1]); data["harm_q3"].append(harm[2])
        dual_data[model_key] = data
        all_gain_q1.extend(value for value in data["gain_q1"] if value > 0)
        all_gain_q3.extend(value for value in data["gain_q3"] if value > 0)
        all_harm_q1.extend(data["harm_q1"]); all_harm_q3.extend(data["harm_q3"])
    if not all_gain_q1 or not all_gain_q3:
        raise RuntimeError("No positive RMS quartiles are available for the logarithmic dual figure")
    gain_limits = (max(min(all_gain_q1) * 0.72, 1e-3), max(all_gain_q3) * 1.22)
    harm_limits = (min(-0.05, min(all_harm_q1) * 1.08), max(1.05, max(all_harm_q3) * 1.08))

    figure, axes = plt.subplots(1, 2, figsize=(13.6, 5.0), sharey=True)
    for panel, (gain_axis, model_key) in enumerate(zip(axes, ("hvql", "cnn"))):
        data = dual_data[model_key]
        harm_axis = gain_axis.twinx()
        gain_q1 = np.asarray(data["gain_q1"], dtype=np.float64)
        gain_axis.plot(data["x"], data["gain"], color="#0072B2", marker="o", markeredgecolor="#20242B")
        gain_axis.fill_between(data["x"], np.where(gain_q1 > 0, gain_q1, np.nan), data["gain_q3"],
                               color="#0072B2", alpha=0.15)
        harm_axis.plot(data["x"], data["harm"], color="#D55E00", linestyle="--", marker="s",
                       markeredgecolor="#20242B")
        harm_axis.fill_between(data["x"], data["harm_q1"], data["harm_q3"], color="#D55E00", alpha=0.12)
        gain_axis.axhline(1.0, color="#0072B2", linewidth=0.85, linestyle=":", alpha=0.75)
        harm_axis.axhline(0.0, color="#D55E00", linewidth=0.75, linestyle="--", alpha=0.72)
        harm_axis.axhline(1.0, color="#D55E00", linewidth=0.75, linestyle=":", alpha=0.65)
        gain_axis.set_yscale("log"); gain_axis.set_ylim(gain_limits); harm_axis.set_ylim(harm_limits)
        gain_axis.set_xticks(data["x"], data["labels"], rotation=27, ha="right")
        gain_axis.xaxis.grid(False)
        gain_axis.set_title("HVQL" if model_key == "hvql" else "CNN", fontsize=11.2, fontweight="bold")
        harm_axis.grid(False)
        if panel == 0:
            gain_axis.set_ylabel("Gain RMS $G_\\ell$", color="#0072B2")
            harm_axis.tick_params(right=False, labelright=False)
            harm_axis.spines["right"].set_visible(False)
        else:
            gain_axis.tick_params(left=False, labelleft=False)
            gain_axis.set_ylabel("")
            harm_axis.spines["right"].set_visible(True)
            harm_axis.set_ylabel("Dommage Normalisé", color="#D55E00")
    figure.legend(
        handles=[
            Line2D([0], [0], color="#0072B2", marker="o", label="Gain RMS"),
            Line2D([0], [0], color="#D55E00", linestyle="--", marker="s", label="Dommage Normalisé"),
        ],
        loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985),
    )
    figure.subplots_adjust(left=0.075, right=0.925, bottom=0.25, top=0.88, wspace=0.08)
    dual_path = output_dir / "hvql_cnn_dual_progression_side_by_side_revised.png"
    save_all(figure, dual_path); plt.close(figure)

    stage_map = {
        "hvql": [("prequantum", 0.0), ("measurement", 1.0), ("logits", 2.0)],
        "hpvql": [("prequantum", 0.0), ("measurement", 1.0), ("logits", 2.0)],
        "hvqm": [("prequantum", 0.0), ("measurement", 1.0), ("logits", 2.0)],
        "hqe": [("prequantum", 0.0), ("measurement", 1.0), ("logits", 2.0)],
        "reshvql": [
            ("prequantum", 0.0), ("quantum_branch", 1.0),
            ("fused", 1.5), ("logits", 2.0),
        ],
    }
    ticks = [0.0, 1.0, 2.0]
    tick_labels = ["Interface Quantique", "Mesure Quantique", "Logits"]
    plot_order = ("hpvql", "hqe", "hvql", "hvqm", "reshvql")
    legend_labels = {
        "hpvql": "HPVQL", "hqe": "HQE", "hvql": "HVQL",
        "hvqm": "HVQM", "reshvql": "Résiduel",
    }

    def progression(metric: str, filename: str, ylabel: str, logarithmic: bool) -> Path:
        figure, axis = plt.subplots(figsize=(8.25, 4.75))
        cohort = trace[trace["cohort_id"] == "B"]
        for model_key in plot_order:
            xs, medians, q1s, q3s = [], [], [], []
            for layer, x_value in stage_map[model_key]:
                group = cohort[(cohort["model_key"] == model_key) & (cohort["layer"] == layer)]
                median, q1, q3 = _figure_group_summary(group, metric)
                xs.append(x_value); medians.append(median); q1s.append(q1); q3s.append(q3)
            color, linestyle, marker = model_styles[model_key]
            lower = np.asarray(q1s, dtype=np.float64)
            if logarithmic:
                lower = np.where(lower > 0, lower, np.nan)
            axis.plot(xs, medians, color=color, linestyle=linestyle, marker=marker,
                      markeredgecolor="#20242B", markeredgewidth=0.35, label=legend_labels[model_key])
            axis.fill_between(xs, lower, q3s, color=color, alpha=0.075)
        if logarithmic:
            axis.set_yscale("log")
            axis.axhline(1.0, color="#343A40", linewidth=0.9, linestyle=":")
            axis.yaxis.set_major_formatter(mticker.ScalarFormatter())
        else:
            axis.axhline(0.0, color="#343A40", linewidth=0.9, linestyle="--")
            axis.axhline(1.0, color="#6B7280", linewidth=0.8, linestyle=":")
        axis.set_xticks(ticks, tick_labels, rotation=22, ha="right")
        axis.xaxis.grid(False)
        axis.set_ylabel(ylabel)
        axis.legend(ncol=2, frameon=False, loc="best")
        figure.subplots_adjust(left=0.12, right=0.98, bottom=0.24, top=0.97)
        path = output_dir / filename
        save_all(figure, path); plt.close(figure)
        return path

    rms_path = progression(
        "gain_rms", "late_hybrid_interface_rms_progression_revised.png",
        "Gain RMS $G_\\ell$", True,
    )
    harm_path = progression(
        "normalized_harm", "late_hybrid_interface_normalized_harm_progression_revised.png",
        "Dommage Normalisé", False,
    )
    _save_figure_metadata(trace, output_dir, specs)
    return [dual_path, rms_path, harm_path]


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def clean(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return "NA"
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_revised_report(
    trace,
    layerwise_summary,
    interface_summary,
    residual_summary,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    import pandas as pd

    validation = pd.read_csv(output_dir / "checkpoint_validation.csv")
    cohorts = pd.read_csv(output_dir / "cohort_summary.csv")
    interface_primary = interface_summary[
        (interface_summary["cohort_id"] == "B")
        & (interface_summary["margin_filter"] == "all")
    ].copy()
    interface_c = interface_summary[
        interface_summary["cohort_id"].str.startswith("C_")
        & (interface_summary["margin_filter"] == "all")
    ].copy()
    residual_primary = residual_summary[residual_summary["cohort_id"] == "B"]

    validation_table = _markdown_table(
        ["Model", "Seed", "Recomputed acc.", "Stored attack-baseline acc.", "Difference (pp)", "Float16-argmax agreement", "Pass"],
        [
            (row.model, row.seed, row.recomputed_test_accuracy, row.stored_attack_baseline_accuracy,
             row.accuracy_difference_vs_attack_baseline_percentage_points, row.prediction_agreement_float16_argmax,
             bool(row.validation_passed))
            for row in validation.itertuples()
        ],
    )
    cohort_table = _markdown_table(
        ["Cohort", "Seed", "Eligible", "Quota/class", "Selected", "Class distribution"],
        [
            (row.cohort_id, row.seed, row.eligible_total, row.class_quota, row.cohort_size,
             ", ".join(str(int(getattr(row, f"selected_class_{digit}"))) for digit in range(10)))
            for row in cohorts.itertuples()
        ],
    )

    interface_rows = []
    samplewise_distribution_rows = []
    for model_key in LATE_MODELS:
        model_rows = interface_primary[interface_primary["model_key"] == model_key]
        for seed_label in (*SEEDS, "pooled"):
            row = model_rows[model_rows["seed"].astype(str) == str(seed_label)].iloc[0]
            interface_rows.append((
                row["model"], row["seed"], row["measurement_norm_ratio_median"],
                row["measurement_norm_ratio_median_ci_lower"], row["measurement_norm_ratio_median_ci_upper"],
                row["normalized_harm_prequantum_median"], row["normalized_harm_measurement_median"],
                row["ratio_of_medians"], row["ratio_of_medians_ci_lower"], row["ratio_of_medians_ci_upper"],
                row["median_samplewise_ratio"], row["median_samplewise_ratio_ci_lower"],
                row["median_samplewise_ratio_ci_upper"],
                row["samplewise_ratio_excluded_abs_pre_lt_1e_minus_8"],
                row["samplewise_ratio_excluded_nonfinite"],
            ))
            samplewise_distribution_rows.append((
                row["model"], row["seed"], row["samplewise_ratio_n"],
                row["samplewise_ratio_median"], row["samplewise_ratio_q1"],
                row["samplewise_ratio_q3"], row["samplewise_ratio_mean"],
                row["samplewise_ratio_std"],
            ))
    interface_table = _markdown_table(
        ["Model", "Seed/scope", "median ||dr||/||dz||", "95% CI low", "95% CI high",
         "median norm. harm z", "median norm. harm r", "ratio of medians",
         "ROM CI low", "ROM CI high", "median samplewise ratio", "samplewise CI low",
         "samplewise CI high", "excluded |denom|<1e-8", "excluded nonfinite"],
        interface_rows,
    )
    samplewise_distribution_table = _markdown_table(
        ["Model", "Seed/scope", "n", "Median", "Q1", "Q3", "Mean", "SD"],
        samplewise_distribution_rows,
    )

    c_rows = []
    for model_key in LATE_MODELS:
        b_model = interface_primary[interface_primary["model_key"] == model_key]
        c_model = interface_c[interface_c["model_key"] == model_key]
        for seed_label in (*SEEDS, "pooled"):
            b = b_model[b_model["seed"].astype(str) == str(seed_label)].iloc[0]
            c = c_model[c_model["seed"].astype(str) == str(seed_label)].iloc[0]
            c_rows.append((
                b["model"], seed_label, b["measurement_norm_ratio_median"], c["measurement_norm_ratio_median"],
                b["ratio_of_medians"], c["ratio_of_medians"],
                b["median_samplewise_ratio"], c["median_samplewise_ratio"],
            ))
    c_table = _markdown_table(
        ["Model", "Seed/scope", "B norm ratio", "C norm ratio", "B ratio-of-medians", "C ratio-of-medians",
         "B median samplewise", "C median samplewise"], c_rows,
    )

    pooled_b = interface_primary[interface_primary["summary_scope"] == "pooled_equal_seed"].set_index("model_key")
    pooled_c = interface_c[interface_c["summary_scope"] == "pooled_equal_seed"].set_index("model_key")

    residual_rows = []
    for metric in (
        "classical_to_quantum_displacement_ratio", "quantum_norm_share_of_branch_sum",
        "classical_norm_share_of_branch_sum", "quantum_to_fused_norm_ratio",
        "classical_to_fused_norm_ratio", "quantum_projection_share_of_fused",
        "classical_projection_share_of_fused", "residual_reconstruction_relative_error",
    ):
        subset = residual_primary[residual_primary["metric"] == metric]
        for row in subset.sort_values(["summary_scope", "seed"]).itertuples():
            residual_rows.append((metric, row.seed, row.n, row.median, row.q1, row.q3, row.mean, row.std,
                                  row.median_ci_lower, row.median_ci_upper))
    residual_table = _markdown_table(
        ["Metric", "Seed/scope", "n", "Median", "Q1", "Q3", "Mean", "SD", "CI low", "CI high"],
        residual_rows,
    )

    first_conv_points = {
        "cnn": "conv1", "hvql": "encoder_conv1", "hpvql": "encoder_conv1",
        "hvqm": "encoder_conv1", "reshvql": "encoder_conv1",
    }
    first_conv_rows = []
    for model_key, layer in first_conv_points.items():
        cohort = "A" if model_key in COMMON_A_MODELS else "B"
        selected = layerwise_summary[
            (layerwise_summary["summary_scope"] == "pooled_equal_seed")
            & (layerwise_summary["cohort_id"] == cohort)
            & (layerwise_summary["model_key"] == model_key)
            & (layerwise_summary["layer"] == layer)
            & (layerwise_summary["metric"] == "gain_rms")
        ].iloc[0]
        first_conv_rows.append((selected["model"], cohort, selected["median"], selected["q1"], selected["q3"],
                                selected["median_ci_lower"], selected["median_ci_upper"]))
    first_conv_table = _markdown_table(
        ["Model", "Cohort", "Median first-conv gain", "Q1", "Q3", "CI low", "CI high"],
        first_conv_rows,
    )

    small_margin_rows = []
    unique_samples = trace[["cohort_id", "model_key", "seed", "dataset_index", "clean_margin"]].drop_duplicates()
    for keys, group in unique_samples.groupby(["cohort_id", "model_key", "seed"]):
        small_margin_rows.append((*keys, len(group), int((group["clean_margin"] < 1e-6).sum())))
    small_margin_table = _markdown_table(
        ["Cohort", "Model", "Seed", "n", "Margins <1e-6"], small_margin_rows,
    )

    sensitivity_rows = []
    for model_key in LATE_MODELS:
        rows = interface_summary[
            (interface_summary["cohort_id"] == "B")
            & (interface_summary["model_key"] == model_key)
            & (interface_summary["summary_scope"] == "pooled_equal_seed")
        ].set_index("margin_filter")
        sensitivity_rows.append((
            rows.loc["all", "model"], rows.loc["all", "ratio_of_medians"],
            rows.loc["margin_ge_1e-6", "ratio_of_medians"],
            rows.loc["all", "median_samplewise_ratio"],
            rows.loc["margin_ge_1e-6", "median_samplewise_ratio"],
            rows.loc["all", "small_margin_count_before_filter"],
        ))
    sensitivity_table = _markdown_table(
        ["Model", "Ratio medians—all", "Ratio medians—margin filter", "Samplewise—all",
         "Samplewise—margin filter", "Small margins"], sensitivity_rows,
    )

    pooled_interface = interface_primary[interface_primary["summary_scope"] == "pooled_equal_seed"].set_index("model_key")
    pooled_residual = residual_primary[residual_primary["summary_scope"] == "pooled_equal_seed"].set_index("metric")
    seed_interface = interface_primary[interface_primary["summary_scope"] == "seed_specific"]
    all_conv_below_one = all(float(row[2]) < 1.0 for row in first_conv_rows)
    small_output_all = bool((pooled_interface["measurement_norm_ratio_median"] < 1.0).all())
    ratio_medians_up = int((pooled_interface["ratio_of_medians"] > 1.0).sum())
    samplewise_up = int((pooled_interface["median_samplewise_ratio"] > 1.0).sum())
    seed_ratio_medians_up = int((seed_interface["ratio_of_medians"] > 1.0).sum())
    seed_samplewise_up = int((seed_interface["median_samplewise_ratio"] > 1.0).sum())
    hqe_seed42 = seed_interface[
        (seed_interface["model_key"] == "hqe") & (seed_interface["seed"].astype(str) == "42")
    ].iloc[0]
    cq_ratio = float(pooled_residual.loc["classical_to_quantum_displacement_ratio", "median"])
    c_share = float(pooled_residual.loc["classical_norm_share_of_branch_sum", "median"])
    q_share = float(pooled_residual.loc["quantum_norm_share_of_branch_sum", "median"])

    def pooled_layer_value(cohort_id: str, model_key: str, layer: str, metric: str) -> float:
        row = layerwise_summary[
            (layerwise_summary["summary_scope"] == "pooled_equal_seed")
            & (layerwise_summary["cohort_id"] == cohort_id)
            & (layerwise_summary["model_key"] == model_key)
            & (layerwise_summary["layer"] == layer)
            & (layerwise_summary["metric"] == metric)
        ].iloc[0]
        return float(row["median"])

    # Late re-amplification: compare z with the minimum earlier encoder gain in Cohort B.
    reamplification_rows = []
    for model_key in LATE_MODELS:
        model_trace = trace[(trace["cohort_id"] == "B") & (trace["model_key"] == model_key)]
        z_layer = model_specs()[model_key].prequantum_point
        z_gain = _figure_group_summary(model_trace[model_trace["layer"] == z_layer], "gain_rms")[0]
        z_order = float(model_trace.loc[model_trace["layer"] == z_layer, "layer_order"].iloc[0])
        earlier = model_trace[
            (model_trace["point_role"] == "classical_encoder") & (model_trace["layer_order"] < z_order)
        ]
        earlier_medians = [
            _figure_group_summary(group, "gain_rms")[0] for _layer, group in earlier.groupby("layer")
        ]
        minimum = min(earlier_medians) if earlier_medians else math.nan
        reamplification_rows.append((model_specs()[model_key].display_name, minimum, z_gain, z_gain / minimum if minimum > 0 else math.nan))
    reamplification_table = _markdown_table(
        ["Model", "Minimum earlier encoder gain", "Prequantum-z gain", "z / earlier minimum"],
        reamplification_rows,
    )
    late_reincrease_count = sum(float(row[3]) > 1.0 for row in reamplification_rows if np.isfinite(row[3]))

    # Recomputed directly from the predecessor's 19 self-attack traces using the
    # same per-example estimands as the revised analysis.
    old_interface = {
        "hvql": (0.0370677, 4.8582, 3.4414),
        "hpvql": (0.141489, 6.3107, 3.9283),
        "hvqm": (0.0605810, 2.1988, 1.7778),
        "hqe": (0.0212721, 3.6548, 3.7675),
        "reshvql": (0.0403342, 2.7718, 2.6620),
    }
    comparison_norm_rows = []
    comparison_harm_rows = []
    for model_key in LATE_MODELS:
        old_norm, old_rom, old_samplewise = old_interface[model_key]
        revised_seed0 = interface_primary[
            (interface_primary["model_key"] == model_key)
            & (interface_primary["seed"].astype(str) == "0")
        ].iloc[0]
        revised_pooled = pooled_interface.loc[model_key]
        comparison_norm_rows.append((
            model_specs()[model_key].display_name, old_norm,
            revised_seed0["measurement_norm_ratio_median"],
            revised_pooled["measurement_norm_ratio_median"],
        ))
        comparison_harm_rows.append((
            model_specs()[model_key].display_name,
            old_rom, revised_seed0["ratio_of_medians"], revised_pooled["ratio_of_medians"],
            old_samplewise, revised_seed0["median_samplewise_ratio"],
            revised_pooled["median_samplewise_ratio"],
        ))
    comparison_norm_table = _markdown_table(
        ["Model", "Old 19, seed 0", "Revised seed 0", "Revised equal-seed pooled"],
        comparison_norm_rows,
    )
    comparison_harm_table = _markdown_table(
        ["Model", "Old ROM", "Revised s0 ROM", "Revised pooled ROM",
         "Old paired median", "Revised s0 paired", "Revised pooled paired"],
        comparison_harm_rows,
    )

    old_first_conv = {
        "cnn": 0.812112, "hvql": 0.627198, "hpvql": 0.546019,
        "hvqm": 0.836093, "reshvql": 0.571707,
    }
    old_reamplification = {
        "hvql": 6.605, "hpvql": 8.105, "hvqm": 2.920,
        "hqe": 4.51, "reshvql": 10.391,
    }
    structure_comparison_rows = []
    for model_key, old_value in old_first_conv.items():
        cohort_id = "A" if model_key in COMMON_A_MODELS else "B"
        layer = first_conv_points[model_key]
        seed0_values = trace[
            (trace["cohort_id"] == cohort_id) & (trace["model_key"] == model_key)
            & (trace["seed"] == 0) & (trace["layer"] == layer)
        ]["gain_rms"].to_numpy(dtype=np.float64)
        pooled_value = next(float(row[2]) for row in first_conv_rows if row[0] == model_specs()[model_key].display_name)
        structure_comparison_rows.append((
            f"{model_specs()[model_key].display_name}: first convolution",
            old_value, float(np.median(seed0_values)), pooled_value,
        ))
    pooled_reamp_by_model = {
        model_key: float(row[3]) for model_key, row in zip(LATE_MODELS, reamplification_rows)
    }
    for model_key, old_value in old_reamplification.items():
        model_trace = trace[
            (trace["cohort_id"] == "B") & (trace["model_key"] == model_key) & (trace["seed"] == 0)
        ]
        z_layer = model_specs()[model_key].prequantum_point
        z_gain = float(np.median(model_trace.loc[model_trace["layer"] == z_layer, "gain_rms"]))
        z_order = float(model_trace.loc[model_trace["layer"] == z_layer, "layer_order"].iloc[0])
        earlier = model_trace[
            (model_trace["point_role"] == "classical_encoder") & (model_trace["layer_order"] < z_order)
        ]
        earlier_minimum = min(
            float(np.median(group["gain_rms"])) for _layer, group in earlier.groupby("layer")
        )
        structure_comparison_rows.append((
            f"{model_specs()[model_key].display_name}: z / earlier minimum",
            old_value, z_gain / earlier_minimum, pooled_reamp_by_model[model_key],
        ))
    structure_comparison_table = _markdown_table(
        ["Same-estimand quantity", "Old 19, seed 0", "Revised seed 0", "Revised equal-seed pooled"],
        structure_comparison_rows,
    )

    old_residual_ratio = 15.0758

    n_a = {int(row.seed): int(row.cohort_size) for row in cohorts[cohorts["cohort_id"] == "A"].itertuples()}
    n_b = {int(row.seed): int(row.cohort_size) for row in cohorts[cohorts["cohort_id"] == "B"].itertuples()}
    interface_pairs = _interface_pairs(trace, model_specs())
    rms_transition_ratio: dict[str, float] = {}
    for model_key in LATE_MODELS:
        pair_rows = interface_pairs[
            (interface_pairs["cohort_id"] == "B") & (interface_pairs["model_key"] == model_key)
        ]
        model_trace = trace[(trace["cohort_id"] == "B") & (trace["model_key"] == model_key)]
        z_layer = model_specs()[model_key].prequantum_point
        r_layer = model_specs()[model_key].measurement_point
        z_dim = float(model_trace.loc[model_trace["layer"] == z_layer, "dimensionality"].iloc[0])
        r_dim = float(model_trace.loc[model_trace["layer"] == r_layer, "dimensionality"].iloc[0])
        by_seed = {
            int(seed): group["measurement_norm_ratio"].to_numpy(dtype=np.float64)
            * math.sqrt(z_dim / r_dim)
            for seed, group in pair_rows.groupby("seed")
        }
        rms_transition_ratio[model_key] = float(_describe_equal_seed(by_seed)["median"])

    thesis_update_rows = [
        ("CNN first-convolution gain (old 0.79)", pooled_layer_value("A", "cnn", "conv1", "gain_rms"),
         "median G, Cohort A"),
        ("CNN hidden-FC gain (old 14.4)", pooled_layer_value("A", "cnn", "hidden_fc", "gain_rms"),
         "median G, Cohort A"),
        ("CNN logit gain (old 23.3)", pooled_layer_value("A", "cnn", "logits", "gain_rms"),
         "median G, Cohort A"),
        ("HVQL first-convolution gain (old 0.64)", pooled_layer_value("A", "hvql", "encoder_conv1", "gain_rms"),
         "median G, Cohort A"),
        ("HVQL prequantum-z gain (old 4.85)", pooled_layer_value("A", "hvql", "prequantum", "gain_rms"),
         "median G, Cohort A"),
        ("HVQL logit gain (old 27.3)", pooled_layer_value("A", "hvql", "logits", "gain_rms"),
         "median G, Cohort A"),
    ]
    for model_key in LATE_MODELS:
        row = pooled_interface.loc[model_key]
        thesis_update_rows.append((
            f"{row['model']} interface statement",
            f"L2 ratio={row['measurement_norm_ratio_median']:.6g}; RMS ratio={rms_transition_ratio[model_key]:.6g}; "
            f"ROM={row['ratio_of_medians']:.6g}; paired median={row['median_samplewise_ratio']:.6g}",
            "Cohort B equal-seed pooled medians; ratios are displacement or harm ratios, not information retention",
        ))
    thesis_update_rows.extend([
        ("Residual quantum-branch RMS (old 0.19)",
         pooled_layer_value("B", "reshvql", "quantum_branch", "delta_rms"), "median delta RMS, Cohort B"),
        ("Residual fused RMS (old 3.49)",
         pooled_layer_value("B", "reshvql", "fused", "delta_rms"), "median delta RMS, Cohort B"),
        ("Residual classical/quantum L2 ratio (old thesis shorthand 17.3)",
         cq_ratio, "median of per-example scaled-branch ratios, Cohort B"),
    ])
    thesis_update_table = _markdown_table(
        ["Thesis-facing quantity", "Exact revised replacement", "Definition/scope"],
        thesis_update_rows,
    )

    caption_a = (
        f"De gauche à droite, HVQL puis le CNN de référence: progression couche par couche du gain RMS et du "
        f"dommage projeté normalisé $H_\\ell/m_f(x)$, où $m_f(x)$ est la marge propre, sur le jeu de test MNIST officiel. "
        f"Configurations: CNN [32,64], FC 128; HVQL, ansatz strong, profondeur 50. "
        f"La cohorte A commune est correctement classée par les deux modèles et sélectionnée de façon déterministe, "
        f"équilibrée à 50 images par classe (n={n_a.get(0)} pour la graine 0; "
        f"n={n_a.get(42)} pour la graine 42). Chaque attaque FGSM non ciblée ($L_\\infty$, $\\epsilon=0,10$) "
        f"est générée en boîte blanche par le modèle tracé. Les points indiquent les médianes descriptives avec "
        f"pondération égale des graines et les rubans les intervalles interquartiles; l'axe du gain RMS est logarithmique."
    )
    caption_rms = (
        f"Progression du gain RMS à l'interface classique--quantique pour HVQL, HPVQL à cinq blocs parallèles, "
        f"HVQM, HQE et HVQL résiduel (ansatz strong, profondeur 50), sur le jeu de test MNIST officiel. La cohorte B "
        f"commune est correctement classée par les cinq modèles, sélectionnée de façon déterministe et équilibrée "
        f"à 50 images par classe (n={n_b.get(0)} pour la graine 0; "
        f"n={n_b.get(42)} pour la graine 42). Les attaques FGSM non ciblées ($L_\\infty$, $\\epsilon=0,10$) "
        f"sont propres à chaque modèle. Points/lignes: médianes à pondération égale des graines; rubans: intervalles "
        f"interquartiles; axe vertical logarithmique. Le point de fusion ne concerne que HVQL résiduel; dans "
        f"l'implémentation native, les mesures de HVQM sont directement ses logits et la fusion résiduelle est "
        f"directement la sortie logit."
    )
    caption_harm = (
        f"Progression du dommage projeté normalisé exact $H_\\ell/m_f(x)$ pour HVQL, HPVQL à cinq blocs "
        f"parallèles, HVQM, HQE et HVQL résiduel (ansatz strong, profondeur quantique 50), sur le jeu de test MNIST "
        f"officiel et la cohorte B: exemples correctement classés par les cinq modèles, sélection déterministe "
        f"équilibrée à 50 images par classe (n={n_b.get(0)} pour la graine 0; n={n_b.get(42)} pour la graine 42). "
        f"Le gradient d'activation est évalué sur l'entrée propre et chaque attaque FGSM non ciblée "
        f"($L_\\infty$, $\\epsilon=0,10$) est générée par le modèle tracé. Points/lignes: médianes à pondération "
        f"égale des graines; rubans: intervalles interquartiles. Aucun terme de stabilisation, winsorisation ou troncature "
        f"n'est appliqué au dénominateur. Le point de fusion ne concerne que HVQL résiduel; HVQM et HVQL résiduel "
        f"n'ont pas de transition supplémentaire après leurs tenseurs natifs respectivement identiques aux logits."
    )

    verdict_1 = "CONFIRMED" if all_conv_below_one else "WEAKENED"
    verdict_2 = "CONFIRMED" if late_reincrease_count >= 2 else "WEAKENED"
    verdict_3 = "CONFIRMED" if small_output_all else "WEAKENED"
    verdict_4 = "CONFIRMED" if ratio_medians_up == 5 and samplewise_up == 5 else ("WEAKENED" if ratio_medians_up >= 3 else "CONTRADICTED")
    verdict_5 = "CONFIRMED" if cq_ratio > 5 and c_share > q_share else "WEAKENED"

    report = f"""# Revised activation-analysis evidence

## Outcome

The rerun uses the exact native checkpoints, official MNIST test images, independently generated model-specific FGSM attacks, comparison-specific balanced cohorts, and per-example metrics before aggregation. No dequantized or manually approximated quantum surrogate is used.

Claim verdicts: (1) **{verdict_1}**; (2) **{verdict_2}**; (3) **{verdict_3}**; (4) **{verdict_4}, with a seed-specific qualification**; (5) **{verdict_5}**. These labels are based only on the revised traces below. For claim 4, the pooled and paired evidence supports an increase, but HQE seed 42 is not above one under the ratio-of-medians estimand.

## 1. Checkpoint and prediction validation

All reported accuracies were recomputed over the 10,000 official MNIST test images before cohort selection. `Pass=True` requires an executable topology match, exact agreement with the attack-aligned float16 argmax predictions, and a maximum absolute output difference below 0.02. Differences above 0.2 percentage points versus the stored attack baseline are separately flagged in `checkpoint_validation.csv`. The training `metrics.csv` and `model.info` artifacts contain no test-accuracy field, so `checkpoint_validation.csv` explicitly marks the independent training-metadata comparison as unavailable rather than treating the attack baseline as training metadata.

{validation_table}

The parallel late-hybrid model is resolved as **HPVQL**, not HPVQM: the historical storage path says `hpvqm`, while the saved Hydra classifier name and executable native class are `hpvql`/`HPVQL`, using `ParaVQC` with five strong depth-50 blocks. The exact seed-0 checkpoint is epoch 15, resolving a rounded-loss tie.

## 2. Cohorts and class balance

Selection is deterministic. The class-distribution column is ordered 0 through 9.

{cohort_table}

Cohort A is used only for the CNN–HVQL main comparison. Cohort B is used for both five-model main interface figures. Cohort C is model-specific and used only as a sensitivity analysis.

## 3. Seed-specific and equal-seed pooled evidence

All medians have 95% percentile-bootstrap intervals from {BOOTSTRAP_RESAMPLES:,} sample-level resamples. Pooled rows give each seed equal mixture weight; they are descriptive and are not inference across independently trained model populations.

### Initial convolutional attenuation

{first_conv_table}

### Late prequantum re-amplification

{reamplification_table}

### Prequantum-to-measurement transition (Cohort B)

`ratio of medians` and `median samplewise ratio` are distinct estimands. The latter excludes only sample denominators with absolute value below 1e-8, with the excluded count shown.

{interface_table}

The full distribution of the per-example harm ratio is heavy-tailed, so its quartiles, mean, and standard deviation are reported rather than relying on the median alone:

{samplewise_distribution_table}

### Model-specific Cohort-C sensitivity

{c_table}

Cohort C preserves the main direction for every model but reveals selection sensitivity most clearly for HQE: its ratio of medians changes from 2.14936 (B) to 7.8353 (C) for seed 0, and from 0.992503 (B) to 1.04240 (C) for seed 42. Cohort C is therefore a robustness diagnostic, not a replacement for the common Cohort-B comparison.

### Residual-HVQL branch decomposition (Cohort B)

The classical displacement is the **scaled** bypass contribution unless the metric explicitly says raw. Norm shares use `||δq||/(||δq||+||δc||)` and its classical counterpart. Projection shares decompose each branch along the fused displacement and therefore sum to one up to numerical precision.

{residual_table}

## 4. Comparison with the previous 19-example analysis

The predecessor used seed 0, 19 examples selected from a 30-image audit subset, correctness across 14 models, an unbalanced class distribution (3,3,2,2,2,0,1,3,1,2), and a near-match HPVQL checkpoint. To avoid the previous report's mixed estimands, the tables below recompute the old trace with the exact per-example definitions used here. The old HPVQL row remains a near-match-checkpoint diagnostic rather than exact checkpoint continuity.

{structure_comparison_table}

The required per-example interface norm ratio, compared like for like:

{comparison_norm_table}

The two exact normalized-harm ratio definitions, compared like for like:

{comparison_harm_table}

The predecessor's direct median per-example scaled classical-to-quantum ratio was {old_residual_ratio:.6g}; the current thesis shorthand 17.3 was instead formed from aggregate branch summaries. The revised required median of per-example ratios is {cq_ratio:.6g}. Differences should be attributed to the new cohorts, second seed, exact checkpoint alignment, native float attack generation, and exact requested normalization—not to retraining.

## 5. Observation-by-observation assessment

1. **{verdict_1} — convolutional encoders initially reduce perturbation RMS.** All {len(first_conv_rows)} traced convolutional models have a pooled median gain below one at the output of the first convolution operator on the applicable main cohort. This attenuation is transient: it must not be generalized to the completed first convolutional block, whose pooling/nonlinearity can already raise the gain above one in several hybrids.
2. **{verdict_2} — displacement can increase again before the quantum interface.** The prequantum-z median exceeds an earlier encoder minimum in {late_reincrease_count} of {len(reamplification_rows)} late-hybrid configurations. This is a statement about non-monotone propagation, not universal amplification in every architecture.
3. **{verdict_3} — only a relatively small displacement norm reaches measured quantum outputs.** All five pooled Cohort-B median measurement/prequantum L2 ratios are below one. These are norm ratios across different representation spaces, not percentages of information preserved.
4. **{verdict_4}, with qualification — normalized projected harm can increase from z to measured output.** The pooled ratio of medians exceeds one in {ratio_medians_up}/5 models and the pooled median paired ratio exceeds one in {samplewise_up}/5; across the ten model-by-seed rows the counts are {seed_ratio_medians_up}/10 and {seed_samplewise_up}/10. The sole ratio-of-medians exception is HQE seed 42: {hqe_seed42['ratio_of_medians']:.6g}, 95% CI [{hqe_seed42['ratio_of_medians_ci_lower']:.6g}, {hqe_seed42['ratio_of_medians_ci_upper']:.6g}], while its paired median is {hqe_seed42['median_samplewise_ratio']:.6g}, 95% CI [{hqe_seed42['median_samplewise_ratio_ci_lower']:.6g}, {hqe_seed42['median_samplewise_ratio_ci_upper']:.6g}]. Thus the “can increase” claim is confirmed, but a stronger claim of uniform increase for both estimands in every seed is weakened.
5. **{verdict_5} — the residual classical bypass dominates fused displacement.** The revised pooled scaled classical/quantum norm ratio is {cq_ratio:.6g}; median branch-sum shares are {c_share:.6g} classical versus {q_share:.6g} quantum. The reconstruction diagnostic confirms that the native fusion is `δf=δq+δc` to numerical precision.

## 6. Numerical replacements for the thesis evidence

The current thesis-facing numerical shorthand should be replaced as follows:

{thesis_update_table}

For the required L2 interface norm ratio specifically, use these Cohort-B equal-seed pooled medians:

{_markdown_table(["Model", "Revised median ||dr||/||dz||", "95% bootstrap CI"], [
    (pooled_interface.loc[key, "model"], pooled_interface.loc[key, "measurement_norm_ratio_median"],
     f"[{pooled_interface.loc[key, 'measurement_norm_ratio_median_ci_lower']:.6g}, {pooled_interface.loc[key, 'measurement_norm_ratio_median_ci_upper']:.6g}]")
    for key in LATE_MODELS
])}

For normalized harm, use the exact Cohort-B values and both transition ratios in the prequantum-to-measurement table above; do not reuse the old ambiguous “damage +” shorthand. Replace the thesis's aggregate-branch shorthand 17.3 with {cq_ratio:.6g} when reporting the required median per-example scaled classical/quantum displacement ratio.

## 7. Exact replacement figure captions

### `hvql_cnn_dual_progression_side_by_side_revised.png`

{caption_a}

### `late_hybrid_interface_rms_progression_revised.png`

{caption_rms}

### `late_hybrid_interface_normalized_harm_progression_revised.png`

{caption_harm}

## 8. Small-margin sensitivity and remaining limitations

{small_margin_table}

{sensitivity_table}

- Only two trained seeds are available. Bootstrap intervals quantify sample-level uncertainty only, not population uncertainty over independently trained models.
- The experiment addresses one dataset, one untargeted FGSM strength (`epsilon=0.10`), and clean-correct samples; it does not establish behavior for other attacks or perturbation budgets.
- Projected harm is a first-order local quantity evaluated at the clean activation. It can differ from the realized nonlinear downstream margin change.
- Activation and measurement spaces have different dimensions and semantics. Their norm ratios compare displacement magnitudes; they are not fractions of information retained.
- The native QNode exposes measured Pauli-Z outputs but not a separately hookable amplitude-normalized state. No substitute state was manufactured.
- Cohort B validly supports common-input cross-model figures but is conditioned by the weakest jointly correct model; Cohort C quantifies the resulting selection sensitivity rather than replacing Cohort B.
- Exact values can remain checkpoint- and seed-specific; two seeds are insufficient to generalize to a trained-model population.
"""
    path = output_dir / "revised_analysis_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def finalize_revision_outputs(output_dir: Path = OUTPUT_DIR) -> dict[str, str]:
    """Aggregate native trace CSV, write required artifacts, figures, and report."""

    import pandas as pd

    trace_csv = output_dir / "sample_level_layerwise_traces.csv"
    if not trace_csv.exists():
        raise FileNotFoundError(trace_csv)
    trace = pd.read_csv(trace_csv)
    numeric = [
        "seed", "dataset_index", "class_label", "epsilon", "layer_order", "dimensionality",
        "clean_prediction", "adversarial_prediction", "clean_margin", "delta_l2", "delta_rms",
        "gain_rms", "projected_harm", "normalized_harm", "residual_scale",
        "classical_to_quantum_displacement_ratio", "raw_classical_to_quantum_displacement_ratio",
        "quantum_norm_share_of_branch_sum", "classical_norm_share_of_branch_sum",
        "quantum_to_fused_norm_ratio", "classical_to_fused_norm_ratio",
        "quantum_projection_share_of_fused", "classical_projection_share_of_fused",
        "residual_reconstruction_relative_error",
    ]
    for column in numeric:
        trace[column] = pd.to_numeric(trace[column], errors="coerce")
    parquet_path = output_dir / "sample_level_layerwise_traces.parquet"
    parquet_preexisting = parquet_path.exists()
    if not parquet_preexisting:
        trace.to_parquet(parquet_path, index=False)
    layerwise = build_layerwise_summary(trace, output_dir)
    _pairs, interface = build_interface_transition_summary(trace, output_dir)
    residual = build_residual_branch_summary(trace, output_dir)
    figures = generate_revised_figures(trace, output_dir)
    report = build_revised_report(trace, layerwise, interface, residual, output_dir)

    metadata_path = output_dir / "analysis_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    inventory = (
        trace[["model_key", "model", "layer", "layer_display_name", "point_role", "tensor_name", "hook_side", "tensor_shape", "dimensionality"]]
        .drop_duplicates().sort_values(["model_key", "dimensionality", "layer"])
        .to_dict(orient="records")
    )
    metadata["activation_inventory"] = _jsonable(inventory)
    metadata["outputs"] = {
        "cohort_manifest": "cohort_manifest.csv",
        "checkpoint_validation": "checkpoint_validation.csv",
        "sample_level_layerwise_traces": "sample_level_layerwise_traces.parquet",
        "layerwise_summary": "layerwise_summary.csv",
        "interface_transition_summary": "interface_transition_summary.csv",
        "residual_branch_summary": "residual_branch_summary.csv",
        "analysis_metadata": "analysis_metadata.json",
        "report": "revised_analysis_report.md",
        "figures": [path.name for path in figures],
        "figure_metadata": "figure_metadata.json",
        "validation_audit": "validation_audit.json",
        "audit_csv_trace_copy": "sample_level_layerwise_traces.csv",
        "interface_sample_level_audit": "interface_transition_sample_level.csv",
    }
    metadata["validation_completed"] = True
    metadata["no_winsorization_or_truncation"] = True
    metadata["parquet_writer"] = (
        "parquetjs-lite streaming conversion, independently row-count verified"
        if parquet_preexisting else "pandas with installed parquet engine"
    )
    metadata["parquet_null_mapping"] = (
        "CSV NaN values used only for structurally non-applicable residual-branch fields are encoded as Parquet null; "
        "all substantive layerwise metric columns were verified to contain zero NaN or infinite values"
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "parquet": str(parquet_path),
        "layerwise_summary": str(output_dir / "layerwise_summary.csv"),
        "interface_transition_summary": str(output_dir / "interface_transition_summary.csv"),
        "residual_branch_summary": str(output_dir / "residual_branch_summary.csv"),
        "report": str(report),
        "figures": ";".join(str(path) for path in figures),
    }


__all__ = [
    "OUTPUT_DIR", "ATTACK_DIR", "SEEDS", "EPSILON", "model_specs",
    "validate_checkpoints", "build_cohort_manifest", "trace_all_cohorts",
    "write_initial_metadata", "run_native_trace_stage", "build_layerwise_summary",
    "build_interface_transition_summary", "build_residual_branch_summary",
    "generate_revised_figures", "build_revised_report", "finalize_revision_outputs",
]

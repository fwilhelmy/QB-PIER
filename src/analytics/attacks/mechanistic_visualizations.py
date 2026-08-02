"""Publication-ready mechanistic figures for adversarial MNIST models.

This module is library-only.  The companion notebook
``mechanistic_visualizations.ipynb`` performs discovery, validation, tracing,
and export.  All publication figures are reconstructed from the CSV written
beside them; no figure depends on an in-memory-only result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import re
from typing import Callable, Iterable, Mapping, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = PROJECT_ROOT / "logs" / "ByModel"
ATTACK_DIR = PROJECT_ROOT / "logs" / "ATTACKS" / "attack_experiment"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_visualizations"
DATA_ROOT = PROJECT_ROOT / "data" / "processed"

SEEDS = (0, 42)
PRIMARY_EPSILON = 0.10
SENSITIVITY_EPSILON = 0.05
TARGET_COHORT = 500
BATCH_SIZE = 100
DIRECTION_BATCH_SIZE = 250
NORMALIZATION_TAU = 1e-3
NORMALIZATION_THRESHOLD = 1e-2
BOOTSTRAP_DRAWS = 1000
DETERMINISTIC_SEED = 20260727


@dataclass(frozen=True)
class BlockSpec:
    module_path: str
    label: str
    label_fr: str
    order: float
    transform: str
    scope: str
    path: str = "main"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    semantic_architecture: str
    raw_attack_name: str
    model_dir: str
    blocks: tuple[BlockSpec, ...]
    interface_pre: str | None = None
    interface_measure: str | None = None
    note: str = ""


PALETTE: Mapping[str, str] = {
    "cnn": "#0072B2",
    "resnet18": "#332288",
    "mlp_small": "#E69F00",
    "mlp_large": "#D55E00",
    "mlpmixer": "#CC79A7",
    "hquanv": "#117733",
    "hvql": "#56B4E9",
    "hpvql5": "#009E73",
    "hpvql10": "#44AA99",
    "hvqm": "#AA4499",
    "hqe": "#999933",
    "reshvql": "#882255",
    "drophvql": "#44AA99",
    "vqm50": "#666666",
}

MARKERS: Mapping[str, str] = {
    "cnn": "o", "resnet18": "s", "mlp_small": "^", "mlp_large": "v",
    "mlpmixer": "D", "hquanv": "P", "hvql": "o", "hpvql5": "s",
    "hpvql10": "^", "hvqm": "D", "hqe": "P", "reshvql": "X",
    "drophvql": "v", "vqm50": "*",
}

LINESTYLES: Mapping[str, str] = {
    "cnn": "-", "resnet18": "--", "mlp_small": "-.", "mlp_large": ":",
    "mlpmixer": "-", "hquanv": "--", "hvql": "-", "hpvql5": "--",
    "hpvql10": ":", "hvqm": "-.", "hqe": ":", "reshvql": "--",
    "drophvql": "-.", "vqm50": ":",
}


def safe_model_name(name: str) -> str:
    return str(name).replace("\\", "__").replace("/", "__").replace(" ", "_").replace(":", "_")


def model_specs() -> dict[str, ModelSpec]:
    conv_rich = (
        BlockSpec("encoder.conv1", "conv1", "Conv. 1", 1, "convolution 5x5", "local"),
        BlockSpec("encoder.maxpool1", "pool1", "Pool 1", 2, "ReLU + max-pool 2x2", "local"),
        BlockSpec("encoder.conv2", "conv2", "Conv. 2", 3, "convolution 5x5", "local/hierarchical"),
        BlockSpec("encoder", "encoder", "Encodeur", 4, "pool + flatten", "hierarchical"),
    )
    late = (
        BlockSpec("classifier.fc1", "interface_dense", "Interface dense", 5, "dense projection", "global/compressive"),
        BlockSpec("classifier.relu", "prequantum", "Latent préquantique", 6, "batch norm + ReLU", "structured latent"),
        BlockSpec("classifier.qlayer", "measurement", "Mesures quantiques", 7, "amplitude encoding + unitary + Pauli-Z measurements", "measured quantum"),
        BlockSpec("classifier.fc2", "head", "Tête classique", 8, "dense readout", "global"),
    )
    return {
        "cnn": ModelSpec(
            "cnn", "CNN", "Classique", "CNN",
            "Classical/convnet/conv_channels=[32,64],fc_sizes=[128]",
            "logs/ByModel/Classical/convnet/conv_channels=[32,64],fc_sizes=[128]",
            (
                BlockSpec("features.0", "conv1", "Conv. 1", 1, "convolution 3x3", "local"),
                BlockSpec("features.2", "pool1", "Pool 1", 2, "ReLU + max-pool", "local"),
                BlockSpec("features.3", "conv2", "Conv. 2", 3, "convolution 3x3", "hierarchical"),
                BlockSpec("features.5", "encoder", "Encodeur", 4, "ReLU + max-pool", "hierarchical"),
                BlockSpec("classifier.0", "dense", "Projection dense", 5, "dense projection", "global"),
                BlockSpec("classifier.1", "predecision", "Pré-décision", 6, "ReLU", "global"),
            ),
        ),
        "resnet18": ModelSpec(
            "resnet18", "ResNet18", "Classique", "ResNet18", "Classical/resnet18",
            "logs/ByModel/Classical/resnet18",
            (
                BlockSpec("conv1", "stem", "Tige conv.", 1, "convolution 3x3", "local"),
                BlockSpec("relu", "stem_relu", "Activation tige", 2, "batch norm + ReLU", "local"),
                BlockSpec("layer1", "res1", "Résiduel 1", 3, "two residual blocks", "hierarchical/residual"),
                BlockSpec("layer2", "res2", "Résiduel 2", 4, "downsample + residual blocks", "hierarchical/residual"),
                BlockSpec("layer3", "res3", "Résiduel 3", 5, "downsample + residual blocks", "hierarchical/residual"),
                BlockSpec("layer4", "encoder", "Encodeur", 6, "downsample + residual blocks", "hierarchical/residual"),
                BlockSpec("avgpool", "predecision", "Moyenne globale", 7, "adaptive average pooling", "global"),
                BlockSpec("linear", "head", "Tête linéaire", 8, "dense readout", "global"),
            ),
        ),
        "mlp_small": ModelSpec(
            "mlp_small", "MLP [128,64]", "Classique", "MLP",
            "Classical/mlp/hidden_sizes=[128,64]",
            "logs/ByModel/Classical/mlp/hidden_sizes=[128,64]",
            (
                BlockSpec("hidden_layers.0", "dense1", "Dense 784→128", 1, "linear output before functional ReLU", "global/raw pixels"),
                BlockSpec("hidden_layers.1", "predecision", "Dense 128→64", 2, "linear output before functional ReLU", "global"),
                BlockSpec("output_layer", "head", "Tête linéaire", 3, "dense readout", "global"),
            ),
        ),
        "mlp_large": ModelSpec(
            "mlp_large", "MLP [1024,512,64,64]", "Classique", "MLP",
            "Classical/mlp/hidden_sizes=[1024, 512, 64, 64]",
            "logs/ByBenchmark/Baseline-Models/Classical/mlp",
            (
                BlockSpec("hidden_layers.0", "dense1", "Dense 784→1024", 1, "linear output before functional ReLU", "global/raw pixels"),
                BlockSpec("hidden_layers.1", "dense2", "Dense 1024→512", 2, "linear output before functional ReLU", "global"),
                BlockSpec("hidden_layers.2", "dense3", "Dense 512→64", 3, "linear output before functional ReLU", "global/compressive"),
                BlockSpec("hidden_layers.3", "predecision", "Dense 64→64", 4, "linear output before functional ReLU", "global"),
                BlockSpec("output_layer", "head", "Tête linéaire", 5, "dense readout", "global"),
            ),
            note="The similarly named ByModel folder contains LeNet and is excluded.",
        ),
        "mlpmixer": ModelSpec(
            "mlpmixer", "MLPMixer", "Classique", "MLPMixer", "Classical/mlpmixer",
            "logs/ByModel/Classical/mlpmixer",
            (
                BlockSpec("patchemb.proj", "patch_projection", "Projection patchs", 1, "stride-4 convolution", "local patches"),
                BlockSpec("patchemb", "patch_tokens", "Jetons patchs", 2, "reshape into tokens", "local patches"),
                BlockSpec("blocks.0", "mixer1", "Mixer 1", 3, "token + channel mixing with residuals", "global tokens"),
                BlockSpec("blocks.1", "mixer2", "Mixer 2", 4, "token + channel mixing with residuals", "global tokens"),
                BlockSpec("blocks.2", "mixer3", "Mixer 3", 5, "token + channel mixing with residuals", "global tokens"),
                BlockSpec("blocks.3", "encoder", "Mixer 4", 6, "token + channel mixing with residuals", "global tokens"),
                BlockSpec("norm", "predecision", "Normalisation", 7, "layer normalization", "global tokens"),
                BlockSpec("head", "head", "Tête linéaire", 8, "mean pool + dense readout", "global"),
            ),
        ),
        "hquanv": ModelSpec(
            "hquanv", "HQuanv", "Hybride précoce", "HQuanv", "Hybrid/hquanv",
            "logs/ByBenchmark/Baseline-Models/Hybrid/hquanv",
            (
                BlockSpec("encoder.quanv", "quanv", "Quanvolution", 1, "shared 2x2 patch circuit", "local patches"),
                BlockSpec("encoder", "encoder", "Carte quanvolution", 2, "flatten", "local patches"),
                BlockSpec("classifier.net.0", "dense", "Projection dense", 3, "dense projection", "global"),
                BlockSpec("classifier.net.1", "predecision", "Pré-décision", 4, "ReLU", "global"),
                BlockSpec("classifier.net.2", "head", "Tête linéaire", 5, "dense readout", "global"),
            ),
        ),
        "hvql": ModelSpec(
            "hvql", "HVQL", "Hybride tardif", "HVQL",
            "Hybrid/hvqm/ansatz=strong,qdepth=50",
            "logs/ByBenchmark/HVQM-Benchmarks/ansatz=strong,qdepth=50",
            conv_rich + late, "prequantum", "measurement",
            "Historical raw folder/attack label says HVQM; executable code is the rich HVQL architecture.",
        ),
        "hpvql5": ModelSpec(
            "hpvql5", "HPVQL (5 blocs)", "Hybride tardif", "HPVQL",
            "Hybrid/hpvqm/qblocks=5,qdepth=50",
            "logs/ByModel/Hybrid/hpvqm/qblocks=5,qdepth=50",
            conv_rich + late, "prequantum", "measurement",
            "Historical raw path says HPVQM; classifier class is executable HPVQL/ParaVQC.",
        ),
        "hpvql10": ModelSpec(
            "hpvql10", "HPVQL (10 blocs)", "Hybride tardif", "HPVQL",
            "Hybrid/hpvqm/qblocks=10,qdepth=50",
            "logs/ByModel/Hybrid/hpvqm/qblocks=10,qdepth=50",
            conv_rich + late, "prequantum", "measurement",
            "Historical raw path says HPVQM; classifier class is executable HPVQL/ParaVQC.",
        ),
        "hvqm": ModelSpec(
            "hvqm", "HVQM", "Hybride tardif", "HVQM",
            "Hybrid/hvql/ansatz=strong,qdepth=50",
            "logs/ByBenchmark/HVQL-Ansatzs/ansatz=strong,qdepth=50",
            (
                BlockSpec("encoder.conv1", "conv1", "Conv. 1", 1, "convolution 5x5", "local"),
                BlockSpec("encoder.maxpool1", "pool1", "Pool 1", 2, "ReLU + max-pool", "local"),
                BlockSpec("encoder.conv2", "conv2", "Conv. 2", 3, "convolution 5x5", "hierarchical"),
                BlockSpec("encoder", "encoder", "Latent préquantique", 4, "pool + flatten", "hierarchical"),
                BlockSpec("classifier.vqc", "measurement", "Mesures quantiques", 5, "amplitude encoding + unitary + Pauli-Z measurements", "measured quantum"),
            ),
            "encoder", "measurement",
            "Historical raw folder/attack label says HVQL; executable code is the small direct-measurement HVQM architecture.",
        ),
        "hqe": ModelSpec(
            "hqe", "HQE", "Hybride tardif", "HQE", "Hybrid/hqe",
            "logs/ByBenchmark/Baseline-Models/Hybrid/hqe",
            (
                BlockSpec("encoder.net.0", "dense4096", "Dense 784→4096", 1, "dense projection", "global/raw pixels"),
                BlockSpec("encoder.net.1", "relu4096", "ReLU 4096", 2, "ReLU", "global"),
                BlockSpec("encoder.net.2", "dense2048", "Dense 4096→2048", 3, "dense compression", "global"),
                BlockSpec("encoder.net.3", "relu2048", "ReLU 2048", 4, "ReLU", "global"),
                BlockSpec("encoder.net.4", "dense1024", "Dense 2048→1024", 5, "dense compression", "global"),
                BlockSpec("encoder.net.5", "encoder", "Latent 1024", 6, "ReLU", "structured latent"),
                BlockSpec("classifier.fc1", "interface_dense", "Interface dense", 7, "dense projection", "global"),
                BlockSpec("classifier.relu", "prequantum", "Latent préquantique", 8, "batch norm + ReLU", "structured latent"),
                BlockSpec("classifier.qlayer", "measurement", "Mesures quantiques", 9, "amplitude encoding + unitary + Pauli-Z measurements", "measured quantum"),
                BlockSpec("classifier.fc2", "head", "Tête classique", 10, "dense readout", "global"),
            ),
            "prequantum", "measurement",
        ),
        "reshvql": ModelSpec(
            "reshvql", "HVQL résiduel", "Hybride tardif", "ResHVQL", "Hybrid/hvql/residuals",
            "logs/ByModel/Hybrid/hvql/residuals",
            conv_rich + (
                BlockSpec("classifier.fc1", "interface_dense", "Interface dense", 5, "dense projection", "global/compressive"),
                BlockSpec("classifier.relu", "prequantum", "Latent préquantique", 6, "batch norm + ReLU", "structured latent"),
                BlockSpec("classifier.vqc", "quantum_branch", "Branche quantique", 7, "VQC measurements", "measured quantum", "quantum"),
                BlockSpec("classifier.residual_proj", "residual_branch", "Contournement classique", 7.1, "dense 1568→10", "global residual", "residual"),
                BlockSpec("classifier", "merge", "Fusion", 8, "quantum + scaled residual", "decision merge", "merge"),
            ),
            "prequantum", "quantum_branch",
        ),
        "drophvql": ModelSpec(
            "drophvql", "HVQL dropout", "Hybride tardif", "DropHVQL", "Hybrid/hvql/dropout",
            "logs/ByModel/Hybrid/hvql/dropout",
            conv_rich + (
                BlockSpec("classifier.fc1", "interface_dense", "Interface dense", 5, "dense projection", "global/compressive"),
                BlockSpec("classifier.relu", "prequantum", "Latent préquantique", 6, "batch norm + ReLU", "structured latent"),
                BlockSpec("classifier.dropout", "dropout", "Dropout (éval.)", 6.5, "identity at inference", "structured latent"),
                BlockSpec("classifier.vqc", "measurement", "Mesures quantiques", 7, "VQC measurements", "measured quantum"),
                BlockSpec("classifier.fc_out", "head", "Tête classique", 8, "dense readout", "global"),
            ),
            "dropout", "measurement",
        ),
        "vqm50": ModelSpec(
            "vqm50", "VQM fort (d=50)", "Quantique", "VQM", "Quantum/VQM50-Strong",
            "logs/ByBenchmark/VQM-Ansatzs/VQM50-Strong",
            (
                BlockSpec("vqc", "measurement", "Mesures quantiques", 1, "amplitude encoding + unitary + Pauli-Z measurements", "global/raw pixel state"),
                BlockSpec("post", "head", "Sortie", 2, "identity output", "decision"),
            ),
            "input", "measurement",
        ),
    }


LAYERWISE_KEYS = ("cnn", "resnet18", "mlp_small", "mlp_large", "mlpmixer", "hquanv", "hvql")
INTERFACE_KEYS = ("hvql", "hpvql5", "hvqm", "hqe")
DIRECTION_SOURCE_KEYS = ("cnn", "mlp_small", "mlp_large", "hquanv", "hvql", "vqm50")
VALIDATION_KEYS = (
    "cnn", "resnet18", "mlp_small", "mlp_large", "mlpmixer", "hquanv",
    "hvql", "hpvql5", "hpvql10", "hvqm", "hqe", "reshvql", "drophvql", "vqm50",
)


def _best_checkpoint(model_dir: Path, seed: int) -> Path:
    candidates = sorted((model_dir / str(seed) / "checkpoints").glob("*.ckpt"))
    if not candidates:
        final = model_dir / str(seed) / "final.ckpt"
        if not final.exists():
            raise FileNotFoundError(final)
        return final

    def score(path: Path) -> float:
        match = re.search(r"loss=(\d+(?:\.\d+)?)", path.name)
        return float(match.group(1)) if match else math.inf

    return min(candidates, key=lambda path: (score(path), path.name))


def load_model(spec: ModelSpec, seed: int):
    from models.ModelFactory import ModelFactory

    model_dir = PROJECT_ROOT / spec.model_dir
    model_name = (model_dir / ".model").read_text(encoding="utf-8").strip()
    checkpoint = _best_checkpoint(model_dir, seed)
    # The attack run selected the second of two checkpoints rounded to the
    # same logged validation loss for HPVQL-5, seed 0.  The filename override
    # is verified against the stored 10,000-image baseline below.
    if spec.key == "hpvql5" and int(seed) == 0:
        checkpoint = model_dir / "0" / "checkpoints" / "epoch=15-loss=0.0327.ckpt"
    model = ModelFactory.load(
        model_name,
        checkpoint=str(checkpoint),
        map_location="cpu",
        log_dir=str(model_dir / str(seed)),
    )
    model.eval()
    return model, checkpoint


def load_test_tensors():
    import torch
    from torchvision.datasets import MNIST

    dataset = MNIST(root=str(DATA_ROOT), train=False, download=False)
    images = dataset.data.to(torch.float32).unsqueeze(1) / 255.0
    labels = dataset.targets.to(torch.long)
    saved = np.load(ATTACK_DIR / "seed_0" / "labels.npy")
    if not np.array_equal(labels.numpy(), saved):
        raise ValueError("Local MNIST test order does not match attack labels.npy")
    return images, labels


def _resolve_module(model, path: str):
    modules = dict(model.named_modules())
    if path not in modules:
        raise KeyError(f"Missing module {path!r}; available: {sorted(modules)}")
    return modules[path]


def _tensor_output(output):
    if hasattr(output, "shape"):
        return output
    if isinstance(output, (tuple, list)) and output and hasattr(output[0], "shape"):
        return output[0]
    raise TypeError(type(output))


def _forward_batched(model, images, batch_size: int = 512):
    import torch

    values = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            values.append(model(images[start : start + batch_size]).detach().to(torch.float32).cpu())
    return torch.cat(values, dim=0)


def validate_checkpoints(
    specs: Mapping[str, ModelSpec] | None = None,
    keys: Sequence[str] = VALIDATION_KEYS,
    seeds: Sequence[int] = SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate executable labels and full-test saved-baseline agreement."""

    import torch

    specs = dict(specs or model_specs())
    images, labels = load_test_tensors()
    alignment_rows: list[dict[str, object]] = []
    architecture_rows: list[dict[str, object]] = []
    for key in tqdm(keys, desc="Validation des points de contrôle"):
        spec = specs[key]
        for seed in seeds:
            model, checkpoint = load_model(spec, seed)
            output = _forward_batched(model, images)
            saved_path = ATTACK_DIR / f"seed_{seed}" / "baselines" / safe_model_name(spec.raw_attack_name) / "logits_clean.npy"
            saved = np.load(saved_path).astype(np.float32)
            saved_preds = np.load(saved_path.parent / "preds_clean.npy")
            error = np.abs(output.numpy() - saved)
            # The attack pipeline cast each forward batch to float16 before
            # argmax.  Reproduce that operation exactly, otherwise near-tied
            # VQM/HVQM logits create false checkpoint mismatches.
            executed_preds = output.to(torch.float16).argmax(1).numpy().astype(np.uint8)
            prediction_match = float(np.mean(executed_preds == saved_preds))
            max_error = float(error.max())
            aligned = bool(prediction_match == 1.0 and max_error < 0.02)
            prediction_dir = OUTPUT_DIR / "validated_predictions"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            np.save(prediction_dir / f"{key}_seed={seed}.npy", executed_preds)
            if key == "mlp_large":
                status = "variante_entraînée_non_attaquée"
                comparison_note = (
                    "The stored attack baseline under the large-MLP raw name belongs to the excluded LeNet copy; "
                    "the discrepancy is expected and the genuine MLP is used only for newly generated traces."
                )
            else:
                status = "aligné" if aligned else "exclure"
                comparison_note = "Exact float16-before-argmax reproduction of the attack pipeline."
            alignment_rows.append({
                "model_key": key,
                "model": spec.label,
                "semantic_architecture": spec.semantic_architecture,
                "raw_attack_name": spec.raw_attack_name,
                "seed": int(seed),
                "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
                "n_samples": int(len(images)),
                "prediction_agreement": prediction_match,
                "max_abs_logit_discrepancy": max_error,
                "mean_abs_logit_discrepancy": float(error.mean()),
                "status": status,
                "comparison_note": comparison_note,
            })
            encoder = type(getattr(model, "encoder", None)).__name__ if hasattr(model, "encoder") else "—"
            classifier = type(getattr(model, "classifier", None)).__name__ if hasattr(model, "classifier") else type(model).__name__
            total_params = int(sum(p.numel() for p in model.parameters()))
            quantum_params = int(sum(
                p.numel() for name, p in model.named_parameters()
                if any(token in name.lower() for token in ("qlayer", "vqc", "ansatz"))
            ))
            measured_dim = None
            if spec.interface_measure:
                module_path = next((b.module_path for b in spec.blocks if b.label == spec.interface_measure), None)
                if module_path:
                    probe: dict[str, object] = {}

                    def probe_hook(_module, _inputs, output):
                        probe["value"] = _tensor_output(output).detach()

                    handle = _resolve_module(model, module_path).register_forward_hook(probe_hook)
                    with torch.no_grad():
                        model(images[:2])
                    handle.remove()
                    measured_dim = int(np.prod(probe["value"].shape[1:]))
            architecture_rows.append({
                "model_key": key,
                "model": spec.label,
                "seed": int(seed),
                "raw_folder_label": spec.raw_attack_name,
                "semantic_architecture": spec.semantic_architecture,
                "model_file": (PROJECT_ROOT / spec.model_dir / ".model").read_text(encoding="utf-8").strip(),
                "executable_class": type(model).__name__,
                "encoder_class": encoder,
                "classifier_class": classifier,
                "total_parameters": total_params,
                "classical_parameters": total_params - quantum_params,
                "quantum_parameters": quantum_params,
                "measured_output_dimension": measured_dim,
                "label_status": "inversion_historique_vérifiée" if key in {"hvql", "hvqm", "hpvql5", "hpvql10"} else "vérifié",
                "note": spec.note,
            })
            del model

    excluded = PROJECT_ROOT / "logs" / "ByModel" / "Classical" / "mlp" / "hidden_sizes=[1024, 512, 64, 64]"
    architecture_rows.append({
        "model_key": "mlp_large_mislabeled",
        "model": "Dossier MLP erroné (exclu)",
        "seed": -1,
        "raw_folder_label": str(excluded.relative_to(PROJECT_ROOT)),
        "semantic_architecture": "LeNet",
        "model_file": (excluded / ".model").read_text(encoding="utf-8").strip(),
        "executable_class": "LeNet",
        "encoder_class": "—",
        "classifier_class": "LeNet",
        "total_parameters": np.nan,
        "classical_parameters": np.nan,
        "quantum_parameters": 0,
        "measured_output_dimension": np.nan,
        "label_status": "exclu",
        "note": "Folder name says MLP [1024,512,64,64], but .model and hparams identify LeNet.",
    })
    return pd.DataFrame(alignment_rows), pd.DataFrame(architecture_rows)


def _balanced_common_correct(
    keys: Sequence[str],
    seed: int,
    specs: Mapping[str, ModelSpec],
    target: int = TARGET_COHORT,
    salt: str = "",
) -> np.ndarray:
    labels = np.load(ATTACK_DIR / f"seed_{seed}" / "labels.npy")
    keep = np.ones(len(labels), dtype=bool)
    for key in keys:
        preds = _cohort_predictions(key, seed, specs)
        keep &= preds == labels
    valid = np.flatnonzero(keep)
    wanted = min(int(target), len(valid))
    digest = int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(DETERMINISTIC_SEED + int(seed) + digest)
    by_class = {c: valid[labels[valid] == c].copy() for c in range(10)}
    for values in by_class.values():
        rng.shuffle(values)
    selected: list[int] = []
    quota = wanted // 10
    for c in range(10):
        selected.extend(by_class[c][:quota].tolist())
        by_class[c] = by_class[c][quota:]
    remainder = wanted - len(selected)
    pool = np.concatenate([by_class[c] for c in range(10)]) if remainder else np.array([], dtype=int)
    if remainder:
        rng.shuffle(pool)
        selected.extend(pool[:remainder].tolist())
    return np.array(sorted(selected), dtype=np.int64)


def _cohort_predictions(key: str, seed: int, specs: Mapping[str, ModelSpec]) -> np.ndarray:
    if key == "mlp_large":
        path = OUTPUT_DIR / "validated_predictions" / f"{key}_seed={seed}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Run validate_checkpoints before cohort construction; missing genuine large-MLP predictions: {path}"
            )
        return np.load(path)
    return np.load(
        ATTACK_DIR / f"seed_{seed}" / "baselines" / safe_model_name(specs[key].raw_attack_name) / "preds_clean.npy"
    )


def build_cohorts(specs: Mapping[str, ModelSpec] | None = None) -> tuple[dict[tuple[str, int], np.ndarray], pd.DataFrame]:
    specs = dict(specs or model_specs())
    groups = {
        "layerwise": LAYERWISE_KEYS,
        "interface": INTERFACE_KEYS,
        "direction": tuple(dict.fromkeys((*INTERFACE_KEYS, *DIRECTION_SOURCE_KEYS))),
        "residual": ("hvql", "reshvql"),
        "segmentation": ("hvql", "hpvql5", "hpvql10"),
    }
    cohorts: dict[tuple[str, int], np.ndarray] = {}
    rows = []
    labels = np.load(ATTACK_DIR / "seed_0" / "labels.npy")
    for group, keys in groups.items():
        for seed in SEEDS:
            indices = _balanced_common_correct(keys, seed, specs, TARGET_COHORT, salt=group)
            cohorts[(group, seed)] = indices
            counts = np.bincount(labels[indices], minlength=10)
            rows.append({
                "cohort": group,
                "seed": seed,
                "models": "; ".join(keys),
                "available_common_correct": int(sum(
                    np.logical_and.reduce([
                        _cohort_predictions(k, seed, specs) == labels
                        for k in keys
                    ])
                )),
                "selected": int(len(indices)),
                "excluded_from_10000": int(10000 - len(indices)),
                **{f"class_{c}": int(counts[c]) for c in range(10)},
            })
    return cohorts, pd.DataFrame(rows)


def _generate_fgsm(model, images, labels, epsilon: float):
    import torch

    x = images.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    output = model(x)
    loss = model.loss_fn(output, labels)
    loss.backward()
    return (x.detach() + float(epsilon) * x.grad.detach().sign()).clamp(0.0, 1.0)


def _matched_random(clean, reference_adv, epsilon: float, seed: int):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    direction = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    direction /= direction.reshape(len(clean), -1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
    target_l2 = (reference_adv - clean).reshape(len(clean), -1).norm(dim=1)
    delta = direction * target_l2.view(-1, 1, 1, 1)
    delta = delta.clamp(-float(epsilon), float(epsilon))
    candidate = (clean + delta).clamp(0.0, 1.0)
    achieved = (candidate - clean).reshape(len(clean), -1).norm(dim=1).clamp_min(1e-12)
    scale = (target_l2 / achieved).clamp(max=8.0)
    for _ in range(4):
        delta = (candidate - clean) * scale.view(-1, 1, 1, 1)
        candidate = (clean + delta.clamp(-float(epsilon), float(epsilon))).clamp(0.0, 1.0)
        achieved = (candidate - clean).reshape(len(clean), -1).norm(dim=1).clamp_min(1e-12)
        scale = target_l2 / achieved
    return candidate


def _capture_clean_margin(model, spec: ModelSpec, images, labels):
    import torch

    captured: dict[str, object] = {}
    handles = []
    for block in spec.blocks:
        module = _resolve_module(model, block.module_path)

        def hook(_module, _inputs, output, key=block.label):
            tensor = _tensor_output(output)
            if tensor.requires_grad:
                tensor.retain_grad()
            captured[key] = tensor

        handles.append(module.register_forward_hook(hook))
    x = images.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    logits.retain_grad()
    row = torch.arange(len(labels))
    competitors = logits.detach().clone()
    competitors[row, labels] = -torch.inf
    competitor = competitors.argmax(1)
    margin = logits[row, labels] - logits[row, competitor]
    margin.sum().backward()
    activations = {"input": x.detach().to(torch.float32).cpu()}
    gradients = {"input": x.grad.detach().to(torch.float32).cpu()}
    for block in spec.blocks:
        tensor = captured[block.label]
        activations[block.label] = tensor.detach().to(torch.float32).cpu()
        gradients[block.label] = tensor.grad.detach().to(torch.float32).cpu() if tensor.grad is not None else None
    activations["logits"] = logits.detach().to(torch.float32).cpu()
    gradients["logits"] = logits.grad.detach().to(torch.float32).cpu()
    for handle in handles:
        handle.remove()
    return activations, gradients, margin.detach().to(torch.float32).cpu(), competitor.detach().cpu()


def _capture_activations(model, spec: ModelSpec, images):
    import torch

    captured: dict[str, object] = {}
    handles = []
    for block in spec.blocks:
        module = _resolve_module(model, block.module_path)
        handles.append(module.register_forward_hook(
            lambda _m, _i, o, key=block.label: captured.__setitem__(key, _tensor_output(o).detach().to(torch.float32).cpu())
        ))
    with torch.no_grad():
        logits = model(images).detach().to(torch.float32).cpu()
    for handle in handles:
        handle.remove()
    return {"input": images.detach().to(torch.float32).cpu(), **captured, "logits": logits}


def _metric_records(
    spec: ModelSpec,
    seed: int,
    cohort: str,
    indices: np.ndarray,
    condition: str,
    source_key: str,
    epsilon: float,
    clean_acts: Mapping[str, object],
    gradients: Mapping[str, object],
    adv_acts: Mapping[str, object],
) -> list[dict[str, object]]:
    import torch

    input_delta = (adv_acts["input"] - clean_acts["input"]).reshape(len(indices), -1)
    input_l2 = input_delta.norm(dim=1)
    input_rms = input_l2 / math.sqrt(input_delta.shape[1])
    grad_input = gradients["input"].reshape(len(indices), -1)
    harm_input = -(grad_input * input_delta).sum(dim=1)
    denom_ok = harm_input.abs() >= NORMALIZATION_THRESHOLD
    blocks = {
        "input": BlockSpec("", "input", "Entrée", 0, "identity", "raw pixels"),
        **{b.label: b for b in spec.blocks},
        "logits": BlockSpec("", "logits", "Logits", 99, "model output", "decision"),
    }
    rows: list[dict[str, object]] = []
    for layer, clean in clean_acts.items():
        adv = adv_acts[layer]
        clean_flat = clean.reshape(len(indices), -1)
        adv_flat = adv.reshape(len(indices), -1)
        delta = adv_flat - clean_flat
        delta_l2 = delta.norm(dim=1)
        delta_rms = delta_l2 / math.sqrt(delta.shape[1])
        grad = gradients.get(layer)
        if grad is None:
            sensitivity = torch.full_like(delta_l2, torch.nan)
            alignment = torch.full_like(delta_l2, torch.nan)
            harm = torch.full_like(delta_l2, torch.nan)
        else:
            grad_flat = grad.reshape(len(indices), -1)
            sensitivity = grad_flat.norm(dim=1)
            alignment = -torch.nn.functional.cosine_similarity(grad_flat, delta, dim=1, eps=1e-12)
            harm = -(grad_flat * delta).sum(dim=1)
        normalized = harm / (harm_input.abs() + NORMALIZATION_TAU)
        normalized = torch.where(denom_ok, normalized, torch.nan)
        normalized_lo = torch.where(denom_ok, harm / (harm_input.abs() + 1e-4), torch.nan)
        normalized_hi = torch.where(denom_ok, harm / (harm_input.abs() + 1e-2), torch.nan)
        decomposition = sensitivity * delta_l2 * alignment
        meta = blocks[layer]
        for pos, dataset_index in enumerate(indices):
            rows.append({
                "model_key": spec.key,
                "model": spec.label,
                "family": spec.family,
                "architecture": spec.semantic_architecture,
                "seed": int(seed),
                "cohort": cohort,
                "dataset_index": int(dataset_index),
                "condition": condition,
                "source_key": source_key,
                "epsilon": float(epsilon),
                "layer": layer,
                "layer_fr": meta.label_fr,
                "layer_order": float(meta.order),
                "path": meta.path,
                "transform": meta.transform,
                "scope": meta.scope,
                "width": int(delta.shape[1]),
                "delta_l2": float(delta_l2[pos]),
                "delta_rms": float(delta_rms[pos]),
                "gain_rms": float(delta_rms[pos] / input_rms[pos].clamp_min(1e-12)),
                "gradient_sensitivity": float(sensitivity[pos]),
                "alignment_harmful": float(alignment[pos]),
                "projected_margin_harm": float(harm[pos]),
                "normalized_margin_harm": float(normalized[pos]),
                "normalized_harm_tau_1e-4": float(normalized_lo[pos]),
                "normalized_harm_tau_1e-2": float(normalized_hi[pos]),
                "decomposition_value": float(decomposition[pos]),
                "decomposition_abs_error": float((decomposition[pos] - harm[pos]).abs()),
                "input_harm": float(harm_input[pos]),
                "normalization_covered": bool(denom_ok[pos]),
            })
    return rows


def _trace_target_condition(
    model,
    spec: ModelSpec,
    seed: int,
    cohort_name: str,
    indices: np.ndarray,
    clean_images,
    labels,
    adv_provider: Callable[[int, int, object, object], object],
    condition: str,
    source_key: str,
    epsilon: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for start in tqdm(range(0, len(indices), BATCH_SIZE), desc=f"{spec.label} · {condition} · ε={epsilon:.2f}", leave=False):
        stop = min(start + BATCH_SIZE, len(indices))
        x = clean_images[start:stop]
        y = labels[start:stop]
        adv = adv_provider(start, stop, x, y)
        clean_acts, gradients, _margin, _competitor = _capture_clean_margin(model, spec, x, y)
        adv_acts = _capture_activations(model, spec, adv)
        records.extend(_metric_records(
            spec, seed, cohort_name, indices[start:stop], condition, source_key, epsilon,
            clean_acts, gradients, adv_acts,
        ))
    return records


def trace_self_layerwise(
    cohorts: Mapping[tuple[str, int], np.ndarray],
    specs: Mapping[str, ModelSpec] | None = None,
    epsilons: Sequence[float] = (SENSITIVITY_EPSILON, PRIMARY_EPSILON),
) -> pd.DataFrame:
    specs = dict(specs or model_specs())
    all_images, all_labels = load_test_tensors()
    records: list[dict[str, object]] = []
    for key in tqdm(LAYERWISE_KEYS, desc="Trajectoires auto-FGSM"):
        spec = specs[key]
        for seed in SEEDS:
            indices = cohorts[("layerwise", seed)]
            images = all_images[indices]
            labels = all_labels[indices]
            model, _ = load_model(spec, seed)
            for epsilon in epsilons:
                provider = lambda _a, _b, x, y, m=model, e=epsilon: _generate_fgsm(m, x, y, e)
                records.extend(_trace_target_condition(
                    model, spec, seed, "layerwise", indices, images, labels,
                    provider, "auto-FGSM", key, float(epsilon),
                ))
            del model
    return pd.DataFrame.from_records(records)


def trace_interfaces(
    cohorts: Mapping[tuple[str, int], np.ndarray],
    specs: Mapping[str, ModelSpec] | None = None,
    epsilons: Sequence[float] = (SENSITIVITY_EPSILON, PRIMARY_EPSILON),
) -> pd.DataFrame:
    specs = dict(specs or model_specs())
    all_images, all_labels = load_test_tensors()
    records: list[dict[str, object]] = []
    for key in tqdm(INTERFACE_KEYS, desc="Interfaces quantiques"):
        spec = specs[key]
        for seed in SEEDS:
            indices = cohorts[("interface", seed)]
            images = all_images[indices]
            labels = all_labels[indices]
            model, _ = load_model(spec, seed)
            for epsilon in epsilons:
                provider = lambda _a, _b, x, y, m=model, e=epsilon: _generate_fgsm(m, x, y, e)
                records.extend(_trace_target_condition(
                    model, spec, seed, "interface", indices, images, labels,
                    provider, "auto-FGSM", key, float(epsilon),
                ))
            del model
    return pd.DataFrame.from_records(records)


def trace_segmentation(
    cohorts: Mapping[tuple[str, int], np.ndarray],
    specs: Mapping[str, ModelSpec] | None = None,
    epsilon: float = PRIMARY_EPSILON,
) -> pd.DataFrame:
    """Trace HVQL and the 5/10-block HPVQL variants on their own paired cohort."""

    specs = dict(specs or model_specs())
    all_images, all_labels = load_test_tensors()
    records: list[dict[str, object]] = []
    for key in tqdm(("hvql", "hpvql5", "hpvql10"), desc="Balayage de segmentation"):
        spec = specs[key]
        for seed in SEEDS:
            indices = cohorts[("segmentation", seed)]
            images = all_images[indices]
            labels = all_labels[indices]
            model, _ = load_model(spec, seed)
            provider = lambda _a, _b, x, y, m=model: _generate_fgsm(m, x, y, epsilon)
            records.extend(_trace_target_condition(
                model, spec, seed, "segmentation", indices, images, labels,
                provider, "auto-FGSM", key, float(epsilon),
            ))
            del model
    return pd.DataFrame.from_records(records)


def _cache_attack_batches(
    keys: Sequence[str],
    seed: int,
    indices: np.ndarray,
    images,
    labels,
    specs: Mapping[str, ModelSpec],
    epsilon: float,
) -> dict[str, object]:
    import torch

    cache: dict[str, object] = {}
    for key in tqdm(keys, desc=f"Directions sources · graine {seed}", leave=False):
        model, _ = load_model(specs[key], seed)
        chunks = []
        for start in range(0, len(indices), DIRECTION_BATCH_SIZE):
            chunks.append(_generate_fgsm(
                model,
                images[start:start + DIRECTION_BATCH_SIZE],
                labels[start:start + DIRECTION_BATCH_SIZE],
                epsilon,
            ))
        cache[key] = torch.cat(chunks, dim=0)
        del model
    return cache


def trace_source_directions(
    cohorts: Mapping[tuple[str, int], np.ndarray],
    specs: Mapping[str, ModelSpec] | None = None,
    epsilon: float = PRIMARY_EPSILON,
) -> pd.DataFrame:
    specs = dict(specs or model_specs())
    all_images, all_labels = load_test_tensors()
    records: list[dict[str, object]] = []
    for seed in SEEDS:
        indices = cohorts[("direction", seed)]
        images = all_images[indices]
        labels = all_labels[indices]
        external = _cache_attack_batches(DIRECTION_SOURCE_KEYS, seed, indices, images, labels, specs, epsilon)
        for target_key in tqdm(INTERFACE_KEYS, desc=f"Cibles hybrides · graine {seed}"):
            spec = specs[target_key]
            model, _ = load_model(spec, seed)
            self_chunks = []
            for start in range(0, len(indices), DIRECTION_BATCH_SIZE):
                self_chunks.append(_generate_fgsm(
                    model,
                    images[start:start + DIRECTION_BATCH_SIZE],
                    labels[start:start + DIRECTION_BATCH_SIZE],
                    epsilon,
                ))
            import torch
            self_adv = torch.cat(self_chunks, dim=0)
            conditions: list[tuple[str, str, object]] = [("auto", target_key, self_adv)]
            conditions.extend(("transfert", source_key, external[source_key]) for source_key in DIRECTION_SOURCE_KEYS)
            random_adv = _matched_random(
                images, self_adv, epsilon,
                DETERMINISTIC_SEED + seed * 100 + sum(map(ord, target_key)),
            )
            conditions.append(("aléatoire", "random", random_adv))
            # Clean activations and clean-margin gradients do not depend on
            # attack source.  Capture them once per target batch, then reuse
            # them for all source directions.
            for start in tqdm(
                range(0, len(indices), DIRECTION_BATCH_SIZE),
                desc=f"{spec.label} · directions",
                leave=False,
            ):
                stop = min(start + DIRECTION_BATCH_SIZE, len(indices))
                x = images[start:stop]
                y = labels[start:stop]
                clean_acts, gradients, _margin, _competitor = _capture_clean_margin(model, spec, x, y)
                for condition, source_key, adversarial in conditions:
                    adv_acts = _capture_activations(model, spec, adversarial[start:stop])
                    records.extend(_metric_records(
                        spec, seed, "direction", indices[start:stop], condition, source_key,
                        float(epsilon), clean_acts, gradients, adv_acts,
                    ))
            del model
    return pd.DataFrame.from_records(records)


def trace_residual_branches(
    cohorts: Mapping[tuple[str, int], np.ndarray],
    specs: Mapping[str, ModelSpec] | None = None,
    epsilon: float = PRIMARY_EPSILON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trace the residual model and export raw/scaled branch contributions."""

    specs = dict(specs or model_specs())
    all_images, all_labels = load_test_tensors()
    rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    spec = specs["reshvql"]
    for seed in SEEDS:
        indices = cohorts[("residual", seed)]
        images = all_images[indices]
        labels = all_labels[indices]
        model, _ = load_model(spec, seed)
        scale = float(model.classifier.residual_scale.detach())
        scale_rows.append({"seed": seed, "trained_residual_scale": scale})
        for start in tqdm(range(0, len(indices), BATCH_SIZE), desc=f"Branches résiduelles · graine {seed}", leave=False):
            stop = min(start + BATCH_SIZE, len(indices))
            x = images[start:stop]
            y = labels[start:stop]
            adv = _generate_fgsm(model, x, y, epsilon)
            clean, gradients, _margin, _competitor = _capture_clean_margin(model, spec, x, y)
            attacked = _capture_activations(model, spec, adv)
            input_delta = (attacked["input"] - clean["input"]).reshape(len(x), -1)
            grad_input = gradients["input"].reshape(len(x), -1)
            input_harm = -(grad_input * input_delta).sum(1)
            denom_ok = input_harm.abs() >= NORMALIZATION_THRESHOLD
            for layer in ("quantum_branch", "residual_branch", "merge"):
                delta = (attacked[layer] - clean[layer]).reshape(len(x), -1)
                grad = gradients[layer].reshape(len(x), -1)
                if layer == "residual_branch":
                    contribution_delta = scale * delta
                    raw_l2 = delta.norm(dim=1)
                else:
                    contribution_delta = delta
                    raw_l2 = delta.norm(dim=1)
                l2 = contribution_delta.norm(dim=1)
                rms = l2 / math.sqrt(contribution_delta.shape[1])
                harm = -(grad * delta).sum(1)
                alignment = -__import__("torch").nn.functional.cosine_similarity(grad, delta, dim=1, eps=1e-12)
                normalized = harm / (input_harm.abs() + NORMALIZATION_TAU)
                normalized = __import__("torch").where(
                    denom_ok, normalized, __import__("torch").full_like(normalized, __import__("torch").nan)
                )
                for pos, dataset_index in enumerate(indices[start:stop]):
                    rows.append({
                        "model_key": "reshvql", "model": spec.label, "seed": seed,
                        "dataset_index": int(dataset_index), "epsilon": float(epsilon),
                        "branch": layer, "branch_fr": {
                            "quantum_branch": "Branche quantique",
                            "residual_branch": "Contournement classique",
                            "merge": "Fusion",
                        }[layer],
                        "trained_residual_scale": scale,
                        "delta_l2_contribution": float(l2[pos]),
                        "delta_rms_contribution": float(rms[pos]),
                        "delta_l2_raw": float(raw_l2[pos]),
                        "projected_margin_harm": float(harm[pos]),
                        "normalized_margin_harm": float(normalized[pos]),
                        "alignment_harmful": float(alignment[pos]),
                        "normalization_covered": bool(denom_ok[pos]),
                    })
        del model
    return pd.DataFrame(rows), pd.DataFrame(scale_rows)


def interface_transitions(trace: pd.DataFrame, specs: Mapping[str, ModelSpec] | None = None) -> pd.DataFrame:
    specs = dict(specs or model_specs())
    rows: list[pd.DataFrame] = []
    for key in trace["model_key"].drop_duplicates():
        spec = specs[key]
        pre = spec.interface_pre
        measured = spec.interface_measure
        frame = trace[trace["model_key"] == key]
        index_cols = ["model_key", "model", "seed", "cohort", "dataset_index", "condition", "source_key", "epsilon"]
        metrics = ["delta_l2", "delta_rms", "projected_margin_harm", "normalized_margin_harm", "alignment_harmful"]
        wide = frame[frame["layer"].isin([pre, measured])].pivot(index=index_cols, columns="layer", values=metrics)
        wide.columns = [f"{metric}__{layer}" for metric, layer in wide.columns]
        wide = wide.reset_index()
        wide["pre_layer"] = pre
        wide["measurement_layer"] = measured
        wide["retention_l2"] = wide[f"delta_l2__{measured}"] / wide[f"delta_l2__{pre}"].clip(lower=1e-12)
        wide["retention_rms"] = wide[f"delta_rms__{measured}"] / wide[f"delta_rms__{pre}"].clip(lower=1e-12)
        wide["harm_change_normalized"] = wide[f"normalized_margin_harm__{measured}"] - wide[f"normalized_margin_harm__{pre}"]
        denominator = wide[f"projected_margin_harm__{pre}"]
        wide["harm_ratio_covered"] = denominator.abs() >= NORMALIZATION_THRESHOLD
        wide["harm_multiplication"] = np.where(
            wide["harm_ratio_covered"],
            wide[f"projected_margin_harm__{measured}"] / denominator,
            np.nan,
        )
        wide["alignment_change"] = wide[f"alignment_harmful__{measured}"] - wide[f"alignment_harmful__{pre}"]
        rows.append(wide)
    return pd.concat(rows, ignore_index=True)


def save_core_tables(
    alignment: pd.DataFrame,
    architecture: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    layerwise: pd.DataFrame,
    interface: pd.DataFrame,
    segmentation: pd.DataFrame,
    directions: pd.DataFrame,
    residual: pd.DataFrame,
    residual_scales: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "checkpoint_alignment": alignment,
        "architecture_label_audit": architecture,
        "cohort_summary": cohort_summary,
        "trace_layerwise_samples": layerwise,
        "trace_interface_samples": interface,
        "trace_segmentation_samples": segmentation,
        "trace_source_direction_samples": directions,
        "trace_residual_samples": residual,
        "residual_scales": residual_scales,
        "interface_transitions": interface_transitions(interface),
        "segmentation_transitions": interface_transitions(segmentation),
        "direction_transitions": interface_transitions(directions),
    }
    paths = {}
    for name, frame in tables.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


COMMON_STAGE_LABELS = {
    0: "Entrée",
    1: "Premier bloc",
    2: "Encodeur profond",
    3: "Pré-interface\n/ décision",
    4: "Mesure",
    5: "Logits",
}

COMMON_STAGE_MAP: Mapping[str, Mapping[str, int]] = {
    "cnn": {"input": 0, "conv1": 1, "encoder": 2, "predecision": 3, "logits": 5},
    "resnet18": {"input": 0, "stem": 1, "encoder": 2, "predecision": 3, "logits": 5},
    "mlp_small": {"input": 0, "dense1": 1, "predecision": 2, "logits": 5},
    "mlp_large": {"input": 0, "dense1": 1, "dense3": 2, "predecision": 3, "logits": 5},
    "mlpmixer": {"input": 0, "patch_projection": 1, "encoder": 2, "predecision": 3, "logits": 5},
    "hquanv": {"input": 0, "quanv": 1, "encoder": 2, "predecision": 3, "logits": 5},
    "hvql": {"input": 0, "conv1": 1, "encoder": 2, "prequantum": 3, "measurement": 4, "logits": 5},
}

SOURCE_LABELS_FR: Mapping[str, str] = {
    "auto": "Auto",
    "cnn": "CNN",
    "mlp_small": "MLP [128,64]",
    "mlp_large": "MLP [1024,512,64,64]",
    "hquanv": "HQuanv",
    "hvql": "HVQL",
    "vqm50": "VQM fort",
    "random": "Aléatoire",
}


def _seed_layer_summary(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    seed = (
        frame.groupby(["model_key", "model", "seed", "layer", "layer_fr", "layer_order"], as_index=False)[metric]
        .agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), n="count")
    )
    rows = []
    for keys, group in seed.groupby(["model_key", "model", "layer", "layer_fr", "layer_order"], sort=False):
        model_key, model, layer, layer_fr, layer_order = keys
        row = {
            "model_key": model_key, "model": model, "layer": layer, "layer_fr": layer_fr,
            "layer_order": layer_order, "estimate": float(group["median"].median()),
            "lower": float(group["q25"].median()), "upper": float(group["q75"].median()),
            "n_samples": int(group["n"].sum()), "n_seeds": int(group["seed"].nunique()),
        }
        for _, seed_row in group.iterrows():
            row[f"seed_{int(seed_row['seed'])}_median"] = float(seed_row["median"])
        rows.append(row)
    return pd.DataFrame(rows)


def common_trajectory_data(
    trace: pd.DataFrame,
    metric: str,
    epsilon: float = PRIMARY_EPSILON,
) -> pd.DataFrame:
    frame = trace[(trace["condition"] == "auto-FGSM") & np.isclose(trace["epsilon"], epsilon)].copy()
    summary = _seed_layer_summary(frame, metric)
    rows = []
    for _, row in summary.iterrows():
        stage = COMMON_STAGE_MAP.get(row["model_key"], {}).get(row["layer"])
        if stage is None:
            continue
        values = row.to_dict()
        values.update({
            "kind": "common_trajectory",
            "metric": metric,
            "stage_index": int(stage),
            "stage_fr": COMMON_STAGE_LABELS[int(stage)],
            "epsilon": float(epsilon),
        })
        rows.append(values)
    return pd.DataFrame(rows).sort_values(["model_key", "stage_index"])


def per_model_trajectory_data(
    trace: pd.DataFrame,
    model_key: str,
    metric: str,
    epsilon: float = PRIMARY_EPSILON,
) -> pd.DataFrame:
    frame = trace[
        (trace["model_key"] == model_key)
        & (trace["condition"] == "auto-FGSM")
        & np.isclose(trace["epsilon"], epsilon)
    ]
    summary = _seed_layer_summary(frame, metric).sort_values("layer_order")
    summary["kind"] = "per_model_trajectory"
    summary["metric"] = metric
    summary["epsilon"] = float(epsilon)
    return summary


def _hierarchical_bootstrap(
    frame: pd.DataFrame,
    value: str,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = DETERMINISTIC_SEED,
) -> tuple[float, float, float]:
    values_by_seed = {
        int(model_seed): group[value].dropna().to_numpy(float)
        for model_seed, group in frame.groupby("seed")
    }
    values_by_seed = {k: v for k, v in values_by_seed.items() if len(v)}
    if not values_by_seed:
        return np.nan, np.nan, np.nan
    seed_ids = np.array(sorted(values_by_seed))
    central = float(np.median([np.median(values_by_seed[s]) for s in seed_ids]))
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=float)
    for draw in range(draws):
        selected_seeds = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        seed_estimates = []
        for model_seed in selected_seeds:
            values = values_by_seed[int(model_seed)]
            sample = rng.choice(values, size=len(values), replace=True)
            seed_estimates.append(np.median(sample))
        boot[draw] = np.median(seed_estimates)
    return central, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def interface_summary_data(
    transitions: pd.DataFrame,
    keys: Sequence[str] = INTERFACE_KEYS,
    condition: str = "auto",
    source_key: str | None = None,
    epsilon: float = PRIMARY_EPSILON,
) -> pd.DataFrame:
    frame = transitions[
        transitions["model_key"].isin(keys)
        & (transitions["condition"] == condition)
        & np.isclose(transitions["epsilon"], epsilon)
    ].copy()
    if source_key is not None:
        frame = frame[frame["source_key"] == source_key]
    rows = []
    for model_key, group in frame.groupby("model_key", sort=False):
        row: dict[str, object] = {
            "kind": "interface_summary", "model_key": model_key,
            "model": group["model"].iloc[0], "epsilon": float(epsilon),
            "condition": condition, "source_key": source_key or "",
            "n_samples": int(len(group)), "n_seeds": int(group["seed"].nunique()),
        }
        for metric in ("retention_rms", "retention_l2", "harm_change_normalized", "harm_multiplication", "alignment_change"):
            central, low, high = _hierarchical_bootstrap(
                group, metric, seed=DETERMINISTIC_SEED + sum(map(ord, model_key + metric))
            )
            row[metric] = central
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        row["harm_ratio_coverage"] = float(group["harm_ratio_covered"].mean())
        for model_seed, seed_group in group.groupby("seed"):
            for metric in ("retention_rms", "retention_l2", "harm_change_normalized", "harm_multiplication", "alignment_change"):
                row[f"seed_{int(model_seed)}_{metric}"] = float(seed_group[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


def transition_scatter_data(
    layerwise: pd.DataFrame,
    interface: pd.DataFrame,
    epsilon: float = PRIMARY_EPSILON,
) -> pd.DataFrame:
    pairs = [
        ("hvql", "prequantum", "measurement", "HVQL : préquantique → mesure", "interface"),
        ("hpvql5", "prequantum", "measurement", "HPVQL : préquantique → mesure", "interface"),
        ("hqe", "prequantum", "measurement", "HQE : préquantique → mesure", "interface"),
        ("mlpmixer", "patch_projection", "mixer1", "MLPMixer : patchs → mixer", "layerwise"),
        ("resnet18", "stem", "encoder", "ResNet18 : tige → résiduels", "layerwise"),
    ]
    rows = []
    for model_key, before, after, label, source in pairs:
        table = interface if source == "interface" else layerwise
        condition = "auto-FGSM"
        frame = table[
            (table["model_key"] == model_key)
            & (table["condition"] == condition)
            & np.isclose(table["epsilon"], epsilon)
            & table["layer"].isin([before, after])
        ]
        index = ["seed", "dataset_index"]
        wide = frame.pivot(index=index, columns="layer", values=["delta_rms", "normalized_margin_harm"]).reset_index()
        wide.columns = ["__".join(filter(None, map(str, col))).rstrip("_") for col in wide.columns]
        wide["log_rms_change"] = np.log(
            wide[f"delta_rms__{after}"].clip(lower=1e-12) / wide[f"delta_rms__{before}"].clip(lower=1e-12)
        )
        wide["harm_change"] = wide[f"normalized_margin_harm__{after}"] - wide[f"normalized_margin_harm__{before}"]
        x, xlo, xhi = _hierarchical_bootstrap(wide, "log_rms_change", seed=DETERMINISTIC_SEED + len(rows))
        y, ylo, yhi = _hierarchical_bootstrap(wide, "harm_change", seed=DETERMINISTIC_SEED + 100 + len(rows))
        rows.append({
            "kind": "transition_scatter", "model_key": model_key, "transition_fr": label,
            "before": before, "after": after, "epsilon": float(epsilon),
            "log_rms_change": x, "log_rms_change_ci_low": xlo, "log_rms_change_ci_high": xhi,
            "harm_change": y, "harm_change_ci_low": ylo, "harm_change_ci_high": yhi,
            "n_samples": int(len(wide)), "n_seeds": int(wide["seed"].nunique()),
        })
    return pd.DataFrame(rows)


def direction_heatmap_data(
    transitions: pd.DataFrame,
    metric: str,
    epsilon: float = PRIMARY_EPSILON,
) -> pd.DataFrame:
    frame = transitions[np.isclose(transitions["epsilon"], epsilon)].copy()
    frame["source_plot"] = np.where(frame["condition"] == "auto", "auto", frame["source_key"])
    seed = frame.groupby(["model_key", "model", "seed", "source_plot"], as_index=False)[metric].median()
    rows = []
    for keys, group in seed.groupby(["model_key", "model", "source_plot"], sort=False):
        model_key, model, source = keys
        rows.append({
            "kind": "heatmap", "metric": metric, "model_key": model_key, "model": model,
            "source_key": source, "source_fr": SOURCE_LABELS_FR.get(source, source),
            "value": float(group[metric].median()),
            "seed_0_value": float(group.loc[group["seed"] == 0, metric].iloc[0]) if (group["seed"] == 0).any() else np.nan,
            "seed_42_value": float(group.loc[group["seed"] == 42, metric].iloc[0]) if (group["seed"] == 42).any() else np.nan,
            "n_seeds": int(group["seed"].nunique()), "epsilon": float(epsilon),
        })
    return pd.DataFrame(rows)


def load_attack_aggregates() -> pd.DataFrame:
    tables = ATTACK_DIR / "tables"
    parquet = sorted(tables.glob("*.parquet"))
    csv = sorted(tables.glob("*.csv"))
    files = parquet or csv
    if not files:
        raise FileNotFoundError(tables)
    frames = [pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def attack_outcome_data(
    aggregates: pd.DataFrame,
    keys: Sequence[str],
    specs: Mapping[str, ModelSpec] | None = None,
    epsilon: float = PRIMARY_EPSILON,
) -> pd.DataFrame:
    specs = dict(specs or model_specs())
    rows = []
    for key in keys:
        raw = specs[key].raw_attack_name
        self_rows = aggregates[(aggregates["source_model"] == raw) & (aggregates["target_model"] == raw)]
        for metric, eps in (("clean_accuracy", 0.0), ("robust_accuracy", epsilon), ("whitebox_asr", epsilon)):
            source_metric = "robust_accuracy" if metric != "whitebox_asr" else "ASR"
            selected = self_rows[np.isclose(self_rows["epsilon"], eps)]
            values = selected.groupby("seed", as_index=False)[source_metric].median()
            rows.append({
                "kind": "dot", "model_key": key, "model": specs[key].label,
                "metric": metric, "epsilon": float(eps),
                "value": float(values[source_metric].median()),
                "lower": float(values[source_metric].min()), "upper": float(values[source_metric].max()),
                "seed_0_value": float(values.loc[values["seed"] == 0, source_metric].iloc[0]) if (values["seed"] == 0).any() else np.nan,
                "seed_42_value": float(values.loc[values["seed"] == 42, source_metric].iloc[0]) if (values["seed"] == 42).any() else np.nan,
                "n_seeds": int(values["seed"].nunique()), "n_samples": int(selected["n_samples"].min()),
            })
    return pd.DataFrame(rows)


def transmission_asr_data(
    transitions: pd.DataFrame,
    aggregates: pd.DataFrame,
    specs: Mapping[str, ModelSpec] | None = None,
    epsilon: float = PRIMARY_EPSILON,
) -> tuple[pd.DataFrame, dict[str, float]]:
    from scipy.stats import spearmanr

    specs = dict(specs or model_specs())
    frame = transitions[
        (transitions["condition"] == "transfert")
        & transitions["source_key"].isin(DIRECTION_SOURCE_KEYS)
        & np.isclose(transitions["epsilon"], epsilon)
    ]
    transmission = frame.groupby(["model_key", "seed", "source_key"], as_index=False).agg(
        retention_rms=("retention_rms", "median"),
        harm_measurement=("normalized_margin_harm__measurement", "median"),
        harm_change=("harm_change_normalized", "median"),
        alignment_change=("alignment_change", "median"),
        latent_displacement=("delta_rms__prequantum", "median"),
        cohort_size=("dataset_index", "nunique"),
    )
    rows = []
    for _, item in transmission.iterrows():
        if item["source_key"] == "mlp_large":
            # The historical attack source with this raw name is the excluded
            # LeNet copy, not the genuine large MLP traced here.
            continue
        target_raw = specs[item["model_key"]].raw_attack_name
        source_raw = specs[item["source_key"]].raw_attack_name
        match = aggregates[
            (aggregates["seed"] == item["seed"])
            & (aggregates["source_model"] == source_raw)
            & (aggregates["target_model"] == target_raw)
            & np.isclose(aggregates["epsilon"], epsilon)
        ]
        if match.empty:
            continue
        rows.append({
            "kind": "asr_scatter", "target_key": item["model_key"],
            "target": specs[item["model_key"]].label, "source_key": item["source_key"],
            "source": specs[item["source_key"]].label, "seed": int(item["seed"]),
            "epsilon": float(epsilon), "incoming_asr": float(match["ASR"].iloc[0]),
            "retention_rms": float(item["retention_rms"]),
            "harm_measurement": float(item["harm_measurement"]),
            "harm_change": float(item["harm_change"]),
            "alignment_change": float(item["alignment_change"]),
            "latent_displacement": float(item["latent_displacement"]),
            "trace_cohort_size": int(item["cohort_size"]),
            "asr_n_samples": int(match["n_samples"].iloc[0]),
        })
    result = pd.DataFrame(rows)
    rho = float(spearmanr(result["harm_measurement"], result["incoming_asr"]).statistic)
    rng = np.random.default_rng(DETERMINISTIC_SEED)
    boot = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = []
        for seed in sorted(result["seed"].unique()):
            group = result[result["seed"] == seed]
            sampled.append(group.sample(len(group), replace=True, random_state=int(rng.integers(0, 2**31 - 1))))
        draw = pd.concat(sampled, ignore_index=True)
        value = spearmanr(draw["harm_measurement"], draw["incoming_asr"]).statistic
        if np.isfinite(value):
            boot.append(float(value))
    stats = {
        "rho": rho,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n_pairs": int(len(result)),
    }
    return result, stats


def residual_summary_data(residual: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for branch, group in residual.groupby("branch", sort=False):
        central, low, high = _hierarchical_bootstrap(
            group, metric, seed=DETERMINISTIC_SEED + sum(map(ord, metric + branch))
        )
        rows.append({
            "kind": "bar", "branch": branch, "branch_fr": group["branch_fr"].iloc[0],
            "metric": metric, "value": central, "lower": low, "upper": high,
            "n_samples": int(len(group)), "n_seeds": int(group["seed"].nunique()),
            "seed_0_value": float(group.loc[group["seed"] == 0, metric].median()),
            "seed_42_value": float(group.loc[group["seed"] == 42, metric].median()),
        })
    order = {"quantum_branch": 0, "residual_branch": 1, "merge": 2}
    result = pd.DataFrame(rows)
    result["order"] = result["branch"].map(order)
    return result.sort_values("order")


def normalization_sensitivity_data(interface_trace: pd.DataFrame) -> pd.DataFrame:
    """Summarize interface harm changes for epsilon and tau sensitivity."""

    columns = {
        "tau=1e-4": "normalized_harm_tau_1e-4",
        "tau=1e-3": "normalized_margin_harm",
        "tau=1e-2": "normalized_harm_tau_1e-2",
    }
    specs = model_specs()
    rows = []
    for model_key in INTERFACE_KEYS:
        spec = specs[model_key]
        subset = interface_trace[interface_trace["model_key"] == model_key]
        for epsilon in sorted(subset["epsilon"].unique()):
            eps_frame = subset[
                np.isclose(subset["epsilon"], epsilon)
                & (subset["condition"] == "auto-FGSM")
                & subset["layer"].isin([spec.interface_pre, spec.interface_measure])
            ]
            index = ["seed", "dataset_index"]
            for tau_label, metric in columns.items():
                wide = eps_frame.pivot(index=index, columns="layer", values=metric).reset_index()
                wide["harm_change"] = wide[spec.interface_measure] - wide[spec.interface_pre]
                seed_values = wide.groupby("seed")["harm_change"].median()
                rows.append({
                    "model_key": model_key,
                    "model": spec.label,
                    "epsilon": float(epsilon),
                    "tau": tau_label,
                    "harm_change": float(seed_values.median()),
                    "seed_0_harm_change": float(seed_values.get(0, np.nan)),
                    "seed_42_harm_change": float(seed_values.get(42, np.nan)),
                    "normalization_coverage": float(eps_frame["normalization_covered"].mean()),
                    "n_samples": int(wide["harm_change"].notna().sum()),
                })
    return pd.DataFrame(rows)


def _plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _clean_axes(ax, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#D9DDE3", linewidth=0.65, alpha=0.7)
    ax.set_axisbelow(True)


def _save_triplet(fig, output_stem: Path) -> dict[str, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for suffix in ("pdf", "svg", "png"):
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=360 if suffix == "png" else None, bbox_inches="tight", pad_inches=0.05)
        paths[suffix] = path
    __import__("matplotlib.pyplot").pyplot.close(fig)
    return paths


def plot_trajectory(data: pd.DataFrame, output_stem: Path, ylabel: str, *, log_scale: bool = False, reference: float | None = None) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    kind = data["kind"].iloc[0]
    x_col = "stage_index" if kind == "common_trajectory" else "layer_order"
    label_col = "stage_fr" if kind == "common_trajectory" else "layer_fr"
    for model_key, group in data.groupby("model_key", sort=False):
        group = group.sort_values(x_col)
        x = group[x_col].to_numpy(float)
        color = PALETTE[model_key]
        ax.fill_between(x, group["lower"], group["upper"], color=color, alpha=0.12, linewidth=0)
        for seed in SEEDS:
            column = f"seed_{seed}_median"
            if column in group:
                ax.plot(x, group[column], color=color, alpha=0.30, linewidth=0.85, linestyle=LINESTYLES[model_key])
        ax.plot(
            x, group["estimate"], label=group["model"].iloc[0], color=color,
            marker=MARKERS[model_key], linestyle=LINESTYLES[model_key], markeredgecolor="#222222", markeredgewidth=0.35,
        )
    labels = data.sort_values(x_col).drop_duplicates(x_col)
    ax.set_xticks(labels[x_col], labels[label_col])
    ax.set_xlabel("Étape architecturale")
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    if reference is not None:
        ax.axhline(reference, color="#333333", linewidth=0.9, linestyle="--", zorder=0)
    _clean_axes(ax)
    if data["model_key"].nunique() > 1:
        ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    _save_triplet(fig, output_stem)


def plot_transition_scatter(data: pd.DataFrame, output_stem: Path) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.axvline(0, color="#444444", linewidth=0.8)
    offsets = {
        "hvql": (-112, 26), "hpvql5": (18, 8), "hqe": (8, 5),
        "mlpmixer": (8, 10), "resnet18": (-150, -19),
    }
    short_labels = {
        "hvql": "HVQL : latent → mesure",
        "hpvql5": "HPVQL : latent → mesure",
        "hqe": "HQE : latent → mesure",
        "mlpmixer": "MLPMixer : projection → mixer",
        "resnet18": "ResNet18 : tige → résiduels",
    }
    for _, row in data.iterrows():
        key = row["model_key"]
        xerr = [[row["log_rms_change"] - row["log_rms_change_ci_low"]], [row["log_rms_change_ci_high"] - row["log_rms_change"]]]
        yerr = [[row["harm_change"] - row["harm_change_ci_low"]], [row["harm_change_ci_high"] - row["harm_change"]]]
        ax.errorbar(
            row["log_rms_change"], row["harm_change"], xerr=xerr, yerr=yerr,
            fmt=MARKERS[key], color=PALETTE[key], ecolor=PALETTE[key], capsize=2.5,
            markeredgecolor="#222222", markeredgewidth=0.4, markersize=7,
        )
        ax.annotate(short_labels.get(key, row["transition_fr"]), (row["log_rms_change"], row["harm_change"]),
                    xytext=offsets.get(key, (5, 5)), textcoords="offset points", fontsize=8)
    ax.set_yscale("symlog", linthresh=0.5, linscale=1.0)
    ax.set_xlabel("Variation logarithmique du déplacement RMS")
    ax.set_ylabel(r"Variation du dommage normalisé $\Delta\widetilde{H}$")
    _clean_axes(ax, grid_axis="both")
    fig.tight_layout()
    _save_triplet(fig, output_stem)


def plot_interface_scatter(data: pd.DataFrame, output_stem: Path, *, x_metric: str = "retention_rms") -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.axvline(1, color="#444444", linewidth=0.8, linestyle="--")
    offsets = {"hvql": (6, 8), "hpvql5": (6, -17), "hvqm": (6, -15), "hqe": (6, 5)}
    for _, row in data.iterrows():
        key = row["model_key"]
        x = row[x_metric]
        y = row["harm_change_normalized"]
        ax.errorbar(
            x, y,
            xerr=[[x - row[f"{x_metric}_ci_low"]], [row[f"{x_metric}_ci_high"] - x]],
            yerr=[[y - row["harm_change_normalized_ci_low"]], [row["harm_change_normalized_ci_high"] - y]],
            fmt=MARKERS[key], color=PALETTE[key], capsize=2.5, markersize=8,
            markeredgecolor="#222222", markeredgewidth=0.4,
        )
        ax.annotate(row["model"], (x, y), xytext=offsets.get(key, (6, 6)), textcoords="offset points", fontsize=8.5)
    if x_metric == "retention_rms" and (data[x_metric] > 0).all() and data[x_metric].max() / data[x_metric].min() > 10:
        ax.set_xscale("log")
    if data["harm_change_normalized"].abs().max() > 5:
        ax.set_yscale("symlog", linthresh=0.5, linscale=1.0)
    ax.set_xlabel("Rétention du déplacement RMS (mesure / latent)")
    if x_metric == "retention_l2":
        ax.set_xlabel("Rétention du déplacement L2 (mesure / latent)")
    ax.set_ylabel(r"Concentration du dommage $\Delta\widetilde{H}$")
    _clean_axes(ax, grid_axis="both")
    fig.tight_layout()
    _save_triplet(fig, output_stem)


def plot_interface_arrows(data: pd.DataFrame, output_stem: Path) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.axhline(1, color="#777777", linewidth=0.7, linestyle=":")
    ax.axvline(1, color="#777777", linewidth=0.7, linestyle=":")
    for _, row in data.iterrows():
        key = row["model_key"]
        y = 1 + row["harm_change_normalized"]
        ax.annotate("", xy=(row["retention_rms"], y), xytext=(1, 1),
                    arrowprops={"arrowstyle": "->", "color": PALETTE[key], "lw": 1.8})
        ax.scatter(row["retention_rms"], y, color=PALETTE[key], marker=MARKERS[key], s=50,
                   edgecolor="#222222", linewidth=0.4, label=row["model"])
    ax.set_xlabel("Rétention du déplacement RMS")
    ax.set_ylabel(r"Niveau relatif de dommage $1+\Delta\widetilde{H}$")
    _clean_axes(ax, grid_axis="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_triplet(fig, output_stem)


def plot_distribution(data: pd.DataFrame, output_stem: Path, metric: str, ylabel: str, *, reference: float | None = None) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    order = list(INTERFACE_KEYS)
    values = [data.loc[data["model_key"] == key, metric].dropna().to_numpy() for key in order]
    labels = [data.loc[data["model_key"] == key, "model"].iloc[0] for key in order]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    box = ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.58,
                     medianprops={"color": "#111111", "linewidth": 1.3},
                     whiskerprops={"color": "#555555"}, capprops={"color": "#555555"})
    for patch, key in zip(box["boxes"], order):
        patch.set_facecolor(PALETTE[key])
        patch.set_alpha(0.48)
        patch.set_edgecolor(PALETTE[key])
    if reference is not None:
        ax.axhline(reference, color="#333333", linewidth=0.9, linestyle="--")
    if metric == "harm_change_normalized":
        ax.set_yscale("symlog", linthresh=0.5, linscale=1.0)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=12)
    _clean_axes(ax)
    fig.tight_layout()
    _save_triplet(fig, output_stem)


def plot_heatmap(data: pd.DataFrame, output_stem: Path, metric: str, label: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    _plot_style()
    row_order = [k for k in INTERFACE_KEYS if k in set(data["model_key"])]
    col_order = [k for k in ("auto", *DIRECTION_SOURCE_KEYS, "random") if k in set(data["source_key"])]
    matrix = data.pivot(index="model_key", columns="source_key", values="value").reindex(index=row_order, columns=col_order)
    values = matrix.to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    if metric in {"harm_change_normalized", "alignment_change"}:
        limit = np.nanmax(np.abs(values))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        image = ax.imshow(values, cmap="PuOr_r", norm=norm, aspect="auto")
    else:
        image = ax.imshow(values, cmap="cividis", aspect="auto")
    ax.set_xticks(range(len(col_order)), [SOURCE_LABELS_FR.get(k, k) for k in col_order], rotation=32, ha="right")
    labels = {k: data.loc[data["model_key"] == k, "model"].iloc[0] for k in row_order}
    ax.set_yticks(range(len(row_order)), [labels[k] for k in row_order])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                rgba = image.cmap(image.norm(values[i, j]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                        color="black" if luminance > 0.55 else "white", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label(label)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    _save_triplet(fig, output_stem)


def plot_asr_scatter(data: pd.DataFrame, output_stem: Path, stats: Mapping[str, float]) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    for target_key, group in data.groupby("target_key", sort=False):
        ax.scatter(
            group["harm_measurement"], group["incoming_asr"], s=38,
            color=PALETTE[target_key], marker=MARKERS[target_key], alpha=0.78,
            edgecolor="#222222", linewidth=0.35, label=group["target"].iloc[0],
        )
    ax.text(0.02, 0.97, rf"$\rho_s={stats['rho']:.2f}$  [{stats['ci_low']:.2f}; {stats['ci_high']:.2f}]",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
    ax.set_xlabel(r"Dommage normalisé à la mesure $\widetilde{H}_{mes}$")
    ax.set_ylabel("ASR entrant exact")
    ax.set_ylim(-0.02, 1.02)
    _clean_axes(ax, grid_axis="both")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save_triplet(fig, output_stem)


def plot_dot(data: pd.DataFrame, output_stem: Path, ylabel: str, *, percent: bool = False, log_scale: bool = False) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    data = data.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for index, row in data.iterrows():
        ax.errorbar(
            index, row["value"],
            yerr=[[row["value"] - row["lower"]], [row["upper"] - row["value"]]],
            fmt=MARKERS[row["model_key"]], color=PALETTE[row["model_key"]], capsize=3,
            markersize=7, markeredgecolor="#222222", markeredgewidth=0.4,
        )
    ax.set_xticks(range(len(data)), data["model"], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    if percent:
        from matplotlib.ticker import PercentFormatter
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0, 1.02)
    if log_scale:
        ax.set_yscale("log")
    _clean_axes(ax)
    fig.tight_layout()
    _save_triplet(fig, output_stem)


def plot_bar(data: pd.DataFrame, output_stem: Path, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    colors = ["#56B4E9", "#D55E00", "#666666"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.arange(len(data))
    bars = ax.bar(x, data["value"], color=colors[:len(data)], alpha=0.72, edgecolor="#333333", linewidth=0.6)
    ax.errorbar(x, data["value"],
                yerr=[data["value"] - data["lower"], data["upper"] - data["value"]],
                fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    ax.set_xticks(x, data["branch_fr"], rotation=10)
    ax.set_ylabel(ylabel)
    _clean_axes(ax)
    fig.tight_layout()
    _save_triplet(fig, output_stem)


def plot_selective_projection_schematic(data: pd.DataFrame, output_stem: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    _plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    origin = np.array([0.8, 2.0])
    obs = np.array([2.2, 0.75])
    non = np.array([1.1, 1.35])
    ax.add_patch(FancyArrowPatch(origin, origin + obs, arrowstyle="->", mutation_scale=12, lw=2.3, color="#0072B2"))
    ax.add_patch(FancyArrowPatch(origin, origin + non, arrowstyle="->", mutation_scale=12, lw=1.8, color="#999999", linestyle="--"))
    ax.text(1.45, 1.35, r"$\Delta z_{obs}$", color="#0072B2")
    ax.text(1.0, 3.0, r"$\Delta z_{non\,obs}$", color="#666666")
    ax.text(0.55, 0.55, r"$\Delta z$", fontsize=12)
    unitary = FancyBboxPatch((3.2, 1.15), 1.35, 1.7, boxstyle="round,pad=0.12", fc="#E8F2F8", ec="#0072B2", lw=1.1)
    measure = FancyBboxPatch((5.25, 1.15), 1.35, 1.7, boxstyle="round,pad=0.12", fc="#F5F3E5", ec="#999933", lw=1.1)
    ax.add_patch(unitary)
    ax.add_patch(measure)
    ax.text(3.875, 2.15, r"$U_{\phi}$", ha="center", va="center", fontsize=14)
    ax.text(3.875, 1.65, "unitaire", ha="center", va="center", fontsize=8)
    ax.text(5.925, 2.15, r"$M_Z$", ha="center", va="center", fontsize=14)
    ax.text(5.925, 1.65, r"$\{\langle Z_i\rangle\}$", ha="center", va="center", fontsize=8)
    ax.add_patch(FancyArrowPatch((2.7, 2), (3.2, 2), arrowstyle="->", mutation_scale=12, lw=1.2, color="#333333"))
    ax.add_patch(FancyArrowPatch((4.55, 2), (5.25, 2), arrowstyle="->", mutation_scale=12, lw=1.2, color="#333333"))
    ax.add_patch(FancyArrowPatch((6.6, 2), (8.15, 2.55), arrowstyle="->", mutation_scale=12, lw=2.0, color="#D55E00"))
    ax.add_patch(FancyArrowPatch((8.15, 1.15), (8.15, 3.2), arrowstyle="->", mutation_scale=12, lw=1.6, color="#AA4499"))
    ax.text(7.15, 2.0, r"$\Delta r$", color="#D55E00")
    ax.text(8.3, 3.1, r"$-\nabla_r m_f(x)$", color="#AA4499")
    ax.text(3.1, 3.25, r"$d(\psi,\psi')$ conservée", fontsize=8, color="#333333")
    ax.text(8.0, 0.55, r"$\|\Delta r\| \ll \|\Delta z\|$", fontsize=9)
    fig.tight_layout()
    _save_triplet(fig, output_stem)


def plot_residual_schematic(data: pd.DataFrame, output_stem: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    _plot_style()
    branch = data.set_index("branch")["value"].to_dict()
    maximum = max(branch.get("quantum_branch", 1e-6), branch.get("residual_branch", 1e-6))
    q_width = 0.8 + 4.2 * branch.get("quantum_branch", 0) / maximum
    r_width = 0.8 + 4.2 * branch.get("residual_branch", 0) / maximum
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.4, 2.0, 1.5, 1.0, "Latent\npréquantique"),
        (3.0, 3.4, 1.6, 0.9, "Interface\nquantique"),
        (3.0, 0.7, 1.6, 0.9, "Contournement\nclassique"),
        (6.2, 2.0, 1.2, 1.0, "Fusion"),
        (8.4, 2.0, 1.1, 1.0, "Logits"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10", fc="white", ec="#444444", lw=1))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5)
    arrows = [
        ((1.9, 2.5), (3.0, 3.85), q_width, "#56B4E9"),
        ((1.9, 2.5), (3.0, 1.15), r_width, "#D55E00"),
        ((4.6, 3.85), (6.2, 2.65), q_width, "#56B4E9"),
        ((4.6, 1.15), (6.2, 2.35), r_width, "#D55E00"),
        ((7.4, 2.5), (8.4, 2.5), 2.2, "#555555"),
    ]
    for start, end, width, color in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, lw=width, color=color, alpha=0.82))
    ax.text(2.2, 4.35, "Branche mesurée", color="#0072B2", fontsize=8)
    ax.text(2.1, 0.25, "Branche de contournement", color="#D55E00", fontsize=8)
    fig.tight_layout()
    _save_triplet(fig, output_stem)


def _companion_text(
    figure_id: str,
    concept: str,
    models: str,
    checkpoints: str,
    cohort: str,
    attack: str,
    epsilon: str,
    blocks: str,
    computation: str,
    normalization: str,
    aggregation: str,
    uncertainty: str,
    exclusions: str,
    interpretation: Sequence[str],
    supported: str,
    unsupported: str,
    limitations: str,
) -> str:
    labels = ["Mesure directe", "Motif visible", "Mécanisme compatible", "Explications alternatives", "Ce que la figure ne prouve pas"]
    bullets = "\n".join(f"{index + 1}. **{labels[index]} —** {value}" for index, value in enumerate(interpretation))
    return f"""# Méthode — {figure_id}

## Concept
{concept}

## Modèles et points de contrôle
{models}

{checkpoints}

## Cohorte
{cohort}

## Attaque
- Source : {attack}
- Epsilon : {epsilon}

## Blocs
{blocks}

## Calcul mathématique
{computation}

## Normalisation
{normalization}

## Agrégation
{aggregation}

## Intervalle d'incertitude
{uncertainty}

## Échantillons exclus
{exclusions}

## Interprétation
{bullets}

## Affirmations soutenues
{supported}

## Affirmations non soutenues
{unsupported}

## Limites
{limitations}

## Reproduction exacte
Exécuter `{figure_id}_plot.py`. Le script lit `{figure_id}.csv` et appelle le moteur de rendu versionné `src/analytics/attacks/mechanistic_visualizations.py`.
"""


def _write_figure_bundle(
    figure_id: str,
    data: pd.DataFrame,
    render: Callable[[pd.DataFrame, Path], None],
    caption: str,
    method: str,
    manifest_row: Mapping[str, object],
    output_dir: Path,
) -> dict[str, object]:
    folder = output_dir / figure_id
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / f"{figure_id}.csv"
    data.to_csv(csv_path, index=False)
    render(pd.read_csv(csv_path), folder / figure_id)
    (folder / f"{figure_id}_method.md").write_text(method, encoding="utf-8")
    (folder / f"{figure_id}_caption.tex").write_text(f"\\caption{{{caption}}}\n", encoding="utf-8")
    script = f'''"""Reproduit exactement {figure_id} depuis son CSV."""\nfrom pathlib import Path\nfrom analytics.attacks.mechanistic_visualizations import reproduce_figure\n\nFIGURE_ID = "{figure_id}"\n\ndef main() -> None:\n    reproduce_figure(FIGURE_ID, Path(__file__).resolve().parent)\n\nif __name__ == "__main__":\n    main()\n'''
    (folder / f"{figure_id}_plot.py").write_text(script, encoding="utf-8")
    return {
        "figure_id": figure_id,
        "filename": f"{figure_id}/{figure_id}.pdf",
        **dict(manifest_row),
    }


def _checkpoint_text(alignment: pd.DataFrame, keys: Sequence[str]) -> str:
    rows = alignment[alignment["model_key"].isin(keys)].sort_values(["model_key", "seed"])
    return "\n".join(
        f"- {row.model}, graine {int(row.seed)} : `{row.checkpoint}`; accord des prédictions = {row.prediction_agreement:.6f}; écart logit max = {row.max_abs_logit_discrepancy:.6g}."
        for row in rows.itertuples()
    )


def reproduce_figure(figure_id: str, folder: Path) -> None:
    data = pd.read_csv(folder / f"{figure_id}.csv")
    stem = folder / figure_id
    if figure_id.startswith("fig_a1_gain") or figure_id.startswith("fig_a2_gain"):
        plot_trajectory(data, stem, r"Gain RMS $G_\ell$", log_scale=True, reference=1)
    elif figure_id == "fig_b1_harm_normalized":
        plot_trajectory(data, stem, r"Dommage de marge normalisé $\widetilde{H}_\ell$", reference=0)
    elif figure_id == "fig_b2_alignment":
        plot_trajectory(data, stem, r"Alignement nuisible $C_\ell$", reference=0)
    elif figure_id == "fig_b3_gradient_sensitivity":
        plot_trajectory(data, stem, r"Sensibilité $S_\ell$", log_scale=True)
    elif figure_id == "fig_b4_norm_harm_transition":
        plot_transition_scatter(data, stem)
    elif figure_id.startswith("fig_c1_interface_rms"):
        plot_interface_scatter(data, stem, x_metric="retention_rms")
    elif figure_id == "fig_c1_interface_l2":
        plot_interface_scatter(data, stem, x_metric="retention_l2")
    elif figure_id == "fig_c2_interface_arrows":
        plot_interface_arrows(data, stem)
    elif figure_id == "fig_c3_retention_distribution":
        plot_distribution(data, stem, "retention_rms", "Rétention RMS", reference=1)
    elif figure_id == "fig_c3_harm_distribution":
        plot_distribution(data, stem, "harm_change_normalized", r"$\Delta\widetilde{H}$", reference=0)
    elif figure_id == "fig_c3_alignment_distribution":
        plot_distribution(data, stem, "alignment_change", r"$\Delta C$", reference=0)
    elif figure_id == "fig_c4_selective_projection":
        plot_selective_projection_schematic(data, stem)
    elif figure_id == "fig_d1_direction_retention":
        plot_heatmap(data, stem, "retention_rms", "Rétention RMS")
    elif figure_id == "fig_d2_direction_harm":
        plot_heatmap(data, stem, "harm_change_normalized", r"$\Delta\widetilde{H}$")
    elif figure_id == "fig_d3_direction_alignment":
        plot_heatmap(data, stem, "alignment_change", r"$\Delta C$")
    elif figure_id == "fig_d5_transmission_asr":
        stats_path = folder / f"{figure_id}_stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        plot_asr_scatter(data, stem, stats)
    elif figure_id.startswith("fig_e1_") or figure_id.startswith("fig_f1_"):
        metric = data["metric"].iloc[0]
        labels = {
            "clean_accuracy": "Exactitude propre", "robust_accuracy": "Exactitude robuste (ε = 0,10)",
            "retention_rms": "Rétention RMS", "harm_change_normalized": r"$\Delta\widetilde{H}$",
            "classical_parameters": "Paramètres classiques", "quantum_parameters": "Paramètres quantiques",
            "measured_output_dimension": "Dimension mesurée",
        }
        plot_dot(data, stem, labels.get(metric, metric), percent=metric.endswith("accuracy"), log_scale="parameters" in metric)
    elif figure_id == "fig_g1_residual_l2":
        plot_bar(data, stem, r"Déplacement L2 de contribution")
    elif figure_id == "fig_g1_residual_rms":
        plot_bar(data, stem, r"Déplacement RMS par composante")
    elif figure_id == "fig_g2_residual_harm":
        plot_bar(data, stem, r"Dommage projeté $H$")
    elif figure_id == "fig_g4_residual_routing":
        plot_residual_schematic(data, stem)
    else:
        raise KeyError(f"Unknown figure id: {figure_id}")


def _cohort_text(cohort_summary: pd.DataFrame, name: str) -> tuple[str, str, int]:
    rows = cohort_summary[cohort_summary["cohort"] == name].sort_values("seed")
    selected = ", ".join(f"{int(row.selected)} (graine {int(row.seed)})" for row in rows.itertuples())
    available = ", ".join(f"{int(row.available_common_correct)} (graine {int(row.seed)})" for row in rows.itertuples())
    cohort = (
        f"Exemples MNIST test correctement classés par tous les modèles de la comparaison; "
        f"sélection stratifiée déterministe par classe. Tailles retenues : {selected}. "
        f"Effectifs communs disponibles avant sous-échantillonnage : {available}."
    )
    exclusions = (
        "Sont exclus les exemples non correctement classés par au moins un modèle de la comparaison, "
        "puis les exemples communs excédant la taille cible de 500. Les gradients non finis et les "
        f"cas |H_entrée| < {NORMALIZATION_THRESHOLD:g} sont exclus uniquement des ratios normalisés."
    )
    return cohort, exclusions, int(rows["selected"].min())


def _layerwise_method(
    figure_id: str,
    concept: str,
    keys: Sequence[str],
    alignment: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    metric_text: str,
    interpretation: Sequence[str],
    supported: str,
    unsupported: str,
    limitations: str,
    epsilon: float = PRIMARY_EPSILON,
) -> str:
    specs = model_specs()
    cohort, exclusions, _ = _cohort_text(cohort_summary, "layerwise")
    return _companion_text(
        figure_id, concept,
        ", ".join(specs[key].label for key in keys),
        _checkpoint_text(alignment, keys),
        cohort, "FGSM blanche propre à chaque modèle, calculée sur les tenseurs non quantifiés",
        f"{epsilon:.2f}",
        "Points de contrôle fonctionnels décrits dans `architecture_label_audit.csv`; les sorties linéaires des MLP sont capturées avant leur ReLU fonctionnelle.",
        metric_text,
        f"RMS par dimension; dommage normalisé avec tau = {NORMALIZATION_TAU:g} et seuil |H_entrée| ≥ {NORMALIZATION_THRESHOLD:g}. Les variantes tau=10^-4 et 10^-2 sont conservées dans les traces brutes.",
        "Médiane et quartiles au niveau des exemples dans chaque graine; courbe centrale = médiane des médianes de graines. Les fines courbes montrent chaque graine.",
        "Bande = médiane inter-graines des quartiles intra-graine; aucune interprétation de significativité statistique.",
        exclusions, interpretation, supported, unsupported, limitations,
    )


def _interface_method(
    figure_id: str,
    concept: str,
    keys: Sequence[str],
    alignment: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    cohort_name: str,
    attack: str,
    computation: str,
    interpretation: Sequence[str],
    supported: str,
    unsupported: str,
    limitations: str,
    epsilon: float = PRIMARY_EPSILON,
) -> str:
    specs = model_specs()
    cohort, exclusions, _ = _cohort_text(cohort_summary, cohort_name)
    return _companion_text(
        figure_id, concept,
        ", ".join(specs[key].label for key in keys),
        _checkpoint_text(alignment, keys),
        cohort, attack, f"{epsilon:.2f}",
        "Transition entre la dernière représentation latente réellement fournie au circuit et le vecteur de mesures accessible à la tête; HVQM utilise la sortie d'encodeur comme pré-interface.",
        computation,
        f"RMS par dimension; dommage H = -∇m·Δa; C = -cos(∇m,Δa); H normalisé avec tau = {NORMALIZATION_TAU:g} après exclusion de |H_entrée| < {NORMALIZATION_THRESHOLD:g}.",
        "Mesures appariées par exemple; médianes intra-graine puis médiane des graines.",
        f"IC à 95 % par bootstrap hiérarchique ({BOOTSTRAP_DRAWS} tirages) rééchantillonnant les graines puis les exemples; deux graines limitent l'interprétation inter-entraînement.",
        exclusions, interpretation, supported, unsupported, limitations,
    )


def _full_attack_method(
    figure_id: str,
    concept: str,
    keys: Sequence[str],
    alignment: pd.DataFrame,
    computation: str,
    interpretation: Sequence[str],
    supported: str,
    unsupported: str,
    limitations: str,
) -> str:
    """Build methods text for exact full-test-set attack outcomes."""

    specs = model_specs()
    return _companion_text(
        figure_id, concept,
        ", ".join(specs[key].label for key in keys),
        _checkpoint_text(alignment, keys),
        "Les 10 000 images du jeu de test MNIST, sans sous-échantillonnage; l'exactitude robuste n'est pas conditionnée à la correction propre.",
        "FGSM blanche propre à chaque modèle, lue dans les artefacts d'attaque sauvegardés",
        f"{PRIMARY_EPSILON:.2f}",
        "Entrée et logits finaux; aucune trace intermédiaire n'intervient dans cette métrique.",
        computation,
        "Aucune normalisation de marge; exactitude = moyenne de 1[argmax(logits adversariaux) = y].",
        "Exactitude calculée sur 10 000 images dans chaque graine; point central = médiane des deux graines.",
        "Étendue entre les deux graines; deux graines ne permettent pas un intervalle inter-entraînement stable.",
        "Aucun échantillon exclu. Les logits float16 sauvegardés sont la source de vérité de l'attaque.",
        interpretation, supported, unsupported, limitations,
    )


def _architecture_method(
    figure_id: str,
    concept: str,
    keys: Sequence[str],
    alignment: pd.DataFrame,
    metric_text: str,
    interpretation: Sequence[str],
    supported: str,
    unsupported: str,
    limitations: str,
) -> str:
    """Build methods text for model-structure quantities without a sample cohort."""

    specs = model_specs()
    return _companion_text(
        figure_id, concept,
        ", ".join(specs[key].label for key in keys),
        _checkpoint_text(alignment, keys),
        "Sans cohorte d'images : quantité structurelle extraite directement du modèle exécutable.",
        "Aucune", "Sans objet",
        "Architecture instanciée depuis la configuration Hydra sauvegardée et le checkpoint aligné.",
        metric_text,
        "Aucune normalisation statistique.",
        "Valeur identique entre graines lorsque l'architecture est identique; la graine 0 est affichée après vérification des deux configurations.",
        "Sans intervalle d'incertitude : il s'agit d'un attribut architectural exact.",
        "Aucun échantillon; les modèles non alignés auraient été exclus avant extraction.",
        interpretation, supported, unsupported, limitations,
    )


def _interface_dot(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    return pd.DataFrame({
        "kind": "dot",
        "model_key": summary["model_key"],
        "model": summary["model"],
        "metric": metric,
        "value": summary[metric],
        "lower": summary[f"{metric}_ci_low"],
        "upper": summary[f"{metric}_ci_high"],
        "n_samples": summary["n_samples"],
        "n_seeds": summary["n_seeds"],
    })


def generate_figure_package(
    alignment: pd.DataFrame,
    architecture: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    layerwise: pd.DataFrame,
    interface: pd.DataFrame,
    segmentation: pd.DataFrame,
    directions: pd.DataFrame,
    residual: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Generate independent PDF/SVG/PNG figures and full companions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, object]] = []
    specs = model_specs()
    attack_aggregates = load_attack_aggregates()
    interface_transition = interface_transitions(interface)
    segmentation_transition = interface_transitions(segmentation)
    direction_transition = interface_transitions(directions)

    def add(
        figure_id: str,
        data: pd.DataFrame,
        render: Callable[[pd.DataFrame, Path], None],
        caption: str,
        method: str,
        *, concept: str,
        models: str,
        attack: str,
        epsilon: str,
        cohort_size: int | str,
        metric: str,
        recommended_use: str,
        status: str,
    ) -> None:
        manifests.append(_write_figure_bundle(
            figure_id, data, render, caption, method,
            {
                "concept": concept, "models": models, "attack": attack, "epsilon": epsilon,
                "cohort_size": cohort_size, "metric": metric,
                "recommended_use": recommended_use, "status": status,
            },
            output_dir,
        ))

    layer_cohort, _, layer_n = _cohort_text(cohort_summary, "layerwise")
    gain = common_trajectory_data(layerwise, "gain_rms")
    first = gain[gain["stage_index"] == 1].set_index("model_key")["estimate"]
    reversals = []
    for key, group in gain.groupby("model_key"):
        if key in first.index and first[key] < 1 and group["estimate"].max() > 1:
            reversals.append(specs[key].label)
    reversal_text = ", ".join(reversals) if reversals else "aucun modèle"
    method = _layerwise_method(
        "fig_a1_gain_layerwise", "Trajectoire du gain RMS à travers des étapes fonctionnelles comparables.",
        LAYERWISE_KEYS, alignment, cohort_summary,
        r"G_l = RMS(Δa_l) / RMS(δ); les représentations de dimensions différentes sont comparées par RMS et non par L2 brut.",
        [
            "La figure mesure le gain RMS médian de chaque déplacement de bloc relativement à la perturbation d'entrée.",
            f"Une atténuation au premier bloc suivie d'un gain supérieur à un est observée pour : {reversal_text}.",
            "Le motif est compatible avec une restructuration progressive plutôt qu'avec une atténuation monotone.",
            "La dimension, les non-linéarités, la marge propre et la qualité d'optimisation peuvent également modifier les trajectoires.",
            "La figure ne permet pas d'attribuer causalement les différences à la localité ou au caractère quantique.",
        ],
        "Elle soutient la distinction entre effet du premier bloc et effet cumulé de l'architecture.",
        "Elle ne soutient ni une loi générale d'atténuation convolutionnelle, ni une vulnérabilité causée par le mélange global.",
        "Les étapes communes sont des jalons fonctionnels, pas des opérations identiques; les sorties pré-ReLU des MLP sont explicitement signalées.",
    )
    add(
        "fig_a1_gain_layerwise", gain,
        lambda d, s: plot_trajectory(d, s, r"Gain RMS $G_\ell$", log_scale=True, reference=1),
        "Gain RMS médian du déplacement adversarial au fil de jalons architecturaux fonctionnels, pour une FGSM blanche à $\\varepsilon=0{,}10$. La bande représente l'intervalle interquartile intra-graine agrégé, et les traits fins les deux graines. Une valeur supérieure à un indique un déplacement RMS plus grand qu'à l'entrée.",
        method, concept="gain RMS par bloc", models="; ".join(specs[k].label for k in LAYERWISE_KEYS),
        attack="auto-FGSM", epsilon="0.10", cohort_size=layer_n, metric="median gain_rms",
        recommended_use="corps principal", status="validé",
    )

    gain_eps005 = common_trajectory_data(layerwise, "gain_rms", epsilon=0.05)
    method_eps005 = _layerwise_method(
        "fig_a1_gain_layerwise_eps005",
        "Analyse de sensibilité du gain RMS à une perturbation plus faible.",
        LAYERWISE_KEYS, alignment, cohort_summary,
        r"G_l = RMS(Δa_l) / RMS(δ), avec ε = 0,05.",
        [
            "La figure mesure la même trajectoire RMS que la version principale à ε=0,05.",
            "Les changements de signe ou d'ordre par rapport à ε=0,10 indiquent une dépendance au régime de perturbation.",
            "La stabilité qualitative est compatible avec un mécanisme local autour des données propres.",
            "Le clipping et les non-linéarités peuvent expliquer des différences entre les deux epsilon.",
            "Cette sensibilité ne constitue pas une courbe de robustesse sur tout le balayage.",
        ],
        "Elle teste si la trajectoire principale persiste dans le régime faible demandé.",
        "Elle ne soutient aucune extrapolation au-delà de ε=0,05 et ε=0,10.",
        "Deux niveaux d'epsilon et deux graines seulement.",
        epsilon=0.05,
    )
    add(
        "fig_a1_gain_layerwise_eps005", gain_eps005,
        lambda d, s: plot_trajectory(d, s, r"Gain RMS $G_\ell$", log_scale=True, reference=1),
        "Analyse de sensibilité du gain RMS médian au fil des blocs sous FGSM blanche à $\\varepsilon=0{,}05$. La bande représente les quartiles intra-graine agrégés et les traits fins les deux graines.",
        method_eps005, concept="sensibilité du gain RMS à epsilon", models="; ".join(specs[k].label for k in LAYERWISE_KEYS),
        attack="auto-FGSM", epsilon="0.05", cohort_size=layer_n, metric="median gain_rms",
        recommended_use="annexe", status="validé",
    )

    for key in ("hvql", "resnet18", "mlp_large", "mlp_small"):
        fig_id = f"fig_a2_gain_{key}"
        data = per_model_trajectory_data(layerwise, key, "gain_rms")
        method = _layerwise_method(
            fig_id, f"Trajectoire détaillée du gain RMS pour {specs[key].label}.",
            (key,), alignment, cohort_summary,
            r"G_l = RMS(Δa_l) / RMS(δ), calculé sur chaque exemple avant agrégation.",
            [
                "La figure mesure le gain RMS à chaque bloc exécutable retenu.",
                "La courbe expose les changements successifs sans agréger les micro-couches à travers d'autres architectures.",
                "Les inversions locales sont compatibles avec des projections et rotations successives du déplacement.",
                "Les changements de dimension, d'activation et de sensibilité aval restent des explications concurrentes.",
                "Une trajectoire d'un seul modèle ne prouve pas qu'un type de bloc cause robustesse ou vulnérabilité.",
            ],
            "Elle soutient une lecture bloc par bloc du modèle indiqué.",
            "Elle ne soutient pas une généralisation à toute la famille architecturale.",
            "Deux graines et une seule profondeur/configuration sont disponibles.",
        )
        add(
            fig_id, data,
            lambda d, s: plot_trajectory(d, s, r"Gain RMS $G_\ell$", log_scale=True, reference=1),
            f"Trajectoire détaillée du gain RMS pour {specs[key].label}, sous FGSM blanche à $\\varepsilon=0{{,}}10$. Les traits fins représentent les graines et la bande les quartiles des exemples.",
            method, concept="alternative détaillée du gain", models=specs[key].label,
            attack="auto-FGSM", epsilon="0.10", cohort_size=layer_n, metric="median gain_rms",
            recommended_use="annexe", status="validé",
        )

    harm = common_trajectory_data(layerwise, "normalized_margin_harm")
    method = _layerwise_method(
        "fig_b1_harm_normalized", "Trajectoire signée de la pertinence décisionnelle du déplacement.",
        LAYERWISE_KEYS, alignment, cohort_summary,
        r"H_l = -∇_{a_l}m_f(x)^TΔa_l; Htilde_l = H_l/(|H_input|+tau), après seuil de dénominateur.",
        [
            "La figure mesure le dommage projeté signé dans la direction locale de réduction de marge.",
            "La trajectoire de dommage ne suit pas nécessairement celle du gain RMS.",
            "Cette dissociation est compatible avec une rotation vers des directions plus visibles pour la décision.",
            "Une augmentation de la norme du gradient aval peut produire le même motif sans rotation accrue.",
            "La linéarisation locale H ne remplace pas la marge adversariale finale exacte.",
        ],
        "Elle soutient que magnitude du déplacement et pertinence décisionnelle sont des quantités distinctes.",
        "Elle ne soutient pas que toute baisse de norme augmente le dommage.",
        "H est une projection locale au point propre; les non-linéarités finies à ε=0,10 peuvent créer un écart avec la perte réelle.",
    )
    add(
        "fig_b1_harm_normalized", harm,
        lambda d, s: plot_trajectory(d, s, r"Dommage de marge normalisé $\widetilde{H}_\ell$", reference=0),
        "Dommage de marge projeté et normalisé au fil des blocs sous FGSM blanche à $\\varepsilon=0{,}10$. Le signe est conservé : une valeur positive indique une réduction locale de la marge propre.",
        method, concept="dommage décisionnel par bloc", models="; ".join(specs[k].label for k in LAYERWISE_KEYS),
        attack="auto-FGSM", epsilon="0.10", cohort_size=layer_n, metric="median normalized_margin_harm",
        recommended_use="corps principal", status="validé",
    )

    alignment_data = common_trajectory_data(layerwise, "alignment_harmful")
    method = _layerwise_method(
        "fig_b2_alignment", "Alignement entre déplacement et direction locale nuisible.",
        LAYERWISE_KEYS, alignment, cohort_summary,
        r"C_l = -cos(∇_{a_l}m_f(x), Δa_l), conservant le signe.",
        [
            "La figure mesure l'alignement cosinus avec la direction locale qui réduit la marge.",
            "Des variations d'alignement apparaissent même lorsque le gain RMS varie peu.",
            "Le motif est compatible avec une redirection géométrique du déplacement.",
            "Les changements de gradient dus aux non-linéarités et à la marge propre constituent des explications alternatives.",
            "Un cosinus ne décrit ni la dimension effective complète ni la courbure de la frontière.",
        ],
        "Elle soutient la présence de transformations directionnelles bloc par bloc.",
        "Elle ne prouve pas que ces rotations sont bénéfiques ni spécifiquement quantiques.",
        "L'alignement est calculé au point propre et dépend du compétiteur de classe local.",
    )
    add(
        "fig_b2_alignment", alignment_data,
        lambda d, s: plot_trajectory(d, s, r"Alignement nuisible $C_\ell$", reference=0),
        "Alignement nuisible $C_\\ell=-\\cos(\\nabla m,\\Delta a)$ au fil des blocs, sous FGSM blanche à $\\varepsilon=0{,}10$. Une valeur positive indique une composante orientée vers la réduction locale de marge.",
        method, concept="alignement directionnel", models="; ".join(specs[k].label for k in LAYERWISE_KEYS),
        attack="auto-FGSM", epsilon="0.10", cohort_size=layer_n, metric="median alignment_harmful",
        recommended_use="corps principal ou annexe proche", status="validé",
    )

    sensitivity = common_trajectory_data(layerwise, "gradient_sensitivity")
    method = _layerwise_method(
        "fig_b3_gradient_sensitivity", "Sensibilité locale aval de la marge à chaque représentation.",
        LAYERWISE_KEYS, alignment, cohort_summary,
        r"S_l = ||∇_{a_l}m_f(x)||_2; la décomposition vérifiée est H_l = S_l||Δa_l||_2C_l.",
        [
            "La figure mesure la norme L2 du gradient de marge propre à chaque bloc.",
            "La sensibilité varie fortement avec la représentation et le modèle.",
            "Une hausse de H malgré une baisse de déplacement peut être compatible avec une hausse de S.",
            "Les échelles de paramétrisation et les activations peuvent modifier S sans changement de robustesse intrinsèque.",
            "La norme de gradient seule ne prédit pas l'attaque finie ni le transfert.",
        ],
        "Elle soutient la décomposition du dommage en norme, sensibilité et alignement.",
        "Elle ne soutient pas l'usage de S comme mesure unique de robustesse.",
        "Les valeurs brutes de S ne sont pas directement comparables sans la décomposition complète.",
    )
    add(
        "fig_b3_gradient_sensitivity", sensitivity,
        lambda d, s: plot_trajectory(d, s, r"Sensibilité $S_\ell$", log_scale=True),
        "Sensibilité locale $S_\\ell=\\|\\nabla_{a_\\ell}m_f(x)\\|_2$ au fil des blocs. Les valeurs complètent la magnitude et l'alignement dans $H_\\ell=S_\\ell\\|\\Delta a_\\ell\\|_2C_\\ell$.",
        method, concept="sensibilité de gradient", models="; ".join(specs[k].label for k in LAYERWISE_KEYS),
        attack="auto-FGSM", epsilon="0.10", cohort_size=layer_n, metric="median gradient_sensitivity",
        recommended_use="annexe", status="validé",
    )

    transition_data = transition_scatter_data(layerwise, interface)
    method = _interface_method(
        "fig_b4_norm_harm_transition", "Dissociation entre changement de norme RMS et changement de dommage.",
        ("hvql", "hpvql5", "hqe", "mlpmixer", "resnet18"), alignment, cohort_summary,
        "layerwise", "FGSM blanche propre à chaque modèle, tenseurs non quantifiés",
        r"x = log[RMS(Δa_après)/RMS(Δa_avant)]; y = Htilde_après-Htilde_avant.",
        [
            "Chaque point mesure une transition architecturale préspécifiée sur des exemples appariés.",
            "Les quadrants montrent que baisse de norme et hausse de dommage peuvent coexister.",
            "Ce motif est compatible avec une sensibilité ou un alignement accru après la transition.",
            "La dimension, la compression et la non-linéarité de chaque transition restent des explications concurrentes.",
            "La comparaison ne permet pas d'isoler un effet causal de la mesure quantique.",
        ],
        "Elle soutient directement que changement de magnitude et changement de dommage ne sont pas synonymes.",
        "Elle ne soutient pas qu'une baisse de norme est protectrice ou nuisible en général.",
        "Les transitions sont intentionnellement hétérogènes et servent d'exemples mécanistiques, non de test causal commun.",
    )
    add(
        "fig_b4_norm_harm_transition", transition_data, plot_transition_scatter,
        "Variation du déplacement RMS et variation du dommage de marge normalisé pour cinq transitions architecturales préspécifiées. Les barres indiquent des intervalles bootstrap hiérarchiques à 95\,\%.",
        method, concept="norme versus dommage", models="HVQL; HPVQL; HQE; MLPMixer; ResNet18",
        attack="auto-FGSM", epsilon="0.10", cohort_size=layer_n, metric="log RMS change; normalized harm change",
        recommended_use="corps principal", status="validé",
    )

    interface_summary = interface_summary_data(interface_transition, condition="auto-FGSM")
    compatible = interface_summary[(interface_summary["retention_rms"] < 1) & (interface_summary["harm_change_normalized"] > 0)]["model"].tolist()
    incompatible = interface_summary[~((interface_summary["retention_rms"] < 1) & (interface_summary["harm_change_normalized"] > 0))]["model"].tolist()
    compatible_text = ", ".join(compatible) if compatible else "aucun"
    incompatible_text = ", ".join(incompatible) if incompatible else "aucun"
    c_method = _interface_method(
        "fig_c1_interface_rms", "Sélectivité empirique de l'interface classique–quantique.",
        INTERFACE_KEYS, alignment, cohort_summary, "interface",
        "FGSM blanche propre à chaque cible, calculée sur les tenseurs non quantifiés",
        r"x = médiane hiérarchique de RMS(Δr)/RMS(Δz); y = médiane hiérarchique de Htilde_mesure-Htilde_pré.",
        [
            "La figure mesure simultanément la rétention RMS et le changement de dommage à l'interface.",
            f"Le quadrant rétention < 1 et dommage accru contient : {compatible_text}; autres comportements : {incompatible_text}.",
            "Ce quadrant est compatible avec une transmission directionnelle concentrant la composante nuisible observable.",
            "Une sensibilité accrue de la tête, la compression dimensionnelle et les marges propres peuvent produire un motif analogue.",
            "La figure ne prouve ni un débruitage unitaire ni une causalité spécifiquement quantique.",
        ],
        "Elle soutient le mécanisme sélectif uniquement pour les architectures qui occupent effectivement le quadrant attendu.",
        "Elle ne soutient pas une propriété universelle des couches quantiques; les contre-exemples sont conservés.",
        "Quatre architectures, deux graines et une seule profondeur forte; les IC de graines restent fragiles. Pour HQE à ε=0,10, la normalisation couvre 53,5 % des exemples et varie fortement entre graines; sa coordonnée verticale doit donc être lue comme instable.",
    )
    _, _, interface_n = _cohort_text(cohort_summary, "interface")
    add(
        "fig_c1_interface_rms", interface_summary,
        lambda d, s: plot_interface_scatter(d, s, x_metric="retention_rms"),
        "Rétention RMS du déplacement entre latent préquantique et mesures, confrontée au changement de dommage normalisé, pour quatre hybrides tardifs sous FGSM blanche à $\\varepsilon=0{,}10$. Les barres sont des IC bootstrap hiérarchiques à 95\,\%.",
        c_method, concept="projection quantique sélective", models="HVQL; HPVQL (5); HVQM; HQE",
        attack="auto-FGSM", epsilon="0.10", cohort_size=interface_n, metric="RMS retention; normalized harm change",
        recommended_use="corps principal", status="validé; ordonnée HQE instable",
    )
    interface_summary_eps005 = interface_summary_data(
        interface_transition, condition="auto-FGSM", epsilon=0.05
    )
    c_eps005_method = _interface_method(
        "fig_c1_interface_rms_eps005",
        "Sensibilité de la sélectivité empirique de l'interface à ε=0,05.",
        INTERFACE_KEYS, alignment, cohort_summary, "interface",
        "FGSM blanche propre à chaque cible, calculée sur les tenseurs non quantifiés",
        r"x = médiane hiérarchique de RMS(Δr)/RMS(Δz); y = médiane hiérarchique de Htilde_mesure-Htilde_pré, avec ε=0,05.",
        [
            "La figure répète le diagnostic principal dans le régime FGSM plus faible.",
            "La rétention reste inférieure à un pour les architectures affichées, mais l'ampleur du dommage change.",
            "Une stabilité de quadrant est compatible avec une transmission sélective locale.",
            "Les dénominateurs proches de zéro, surtout pour HQE, et la non-linéarité expliquent aussi des variations.",
            "Cette figure ne transforme pas une association à deux epsilon en relation causale.",
        ],
        "Elle teste la stabilité qualitative du résultat d'interface à faible epsilon.",
        "Elle ne soutient pas la stabilité quantitative de HQE, dont la couverture normalisée n'est que de 37,1 % à ε=0,05.",
        "Deux graines; normalisation HQE particulièrement incomplète; usage d'annexe uniquement.",
        epsilon=0.05,
    )
    add(
        "fig_c1_interface_rms_eps005", interface_summary_eps005,
        lambda d, s: plot_interface_scatter(d, s, x_metric="retention_rms"),
        "Analyse de sensibilité de la rétention RMS et du changement de dommage normalisé à l'interface sous FGSM blanche à $\\varepsilon=0{,}05$. La faible couverture du dommage normalisé pour HQE est explicitement documentée.",
        c_eps005_method, concept="sensibilité de la projection sélective à epsilon", models="HVQL; HPVQL (5); HVQM; HQE",
        attack="auto-FGSM", epsilon="0.05", cohort_size=interface_n, metric="RMS retention; normalized harm change",
        recommended_use="annexe", status="sensibilité; HQE instable",
    )
    c_l2_method = c_method.replace("RMS(Δr)/RMS(Δz)", "||Δr||₂/||Δz||₂").replace("rétention RMS", "rétention L2 brute")
    add(
        "fig_c1_interface_l2", interface_summary,
        lambda d, s: plot_interface_scatter(d, s, x_metric="retention_l2"),
        "Version de sensibilité utilisant la rétention L2 brute à l'interface. La différence avec la version RMS rend visible l'effet de réduction dimensionnelle; cette version est destinée à l'annexe.",
        c_l2_method, concept="alternative L2 de la projection", models="HVQL; HPVQL (5); HVQM; HQE",
        attack="auto-FGSM", epsilon="0.10", cohort_size=interface_n, metric="L2 retention; normalized harm change",
        recommended_use="annexe", status="validé avec réserve dimensionnelle",
    )
    arrow_method = c_method.replace("fig_c1_interface_rms", "fig_c2_interface_arrows")
    add(
        "fig_c2_interface_arrows", interface_summary, plot_interface_arrows,
        "Représentation alternative des transitions d'interface depuis le point normalisé $(1,1)$ vers leur rétention RMS et leur niveau relatif de dommage. Les flèches décrivent des changements empiriques et non une trajectoire d'état quantique.",
        arrow_method, concept="alternative fléchée de l'interface", models="HVQL; HPVQL (5); HVQM; HQE",
        attack="auto-FGSM", epsilon="0.10", cohort_size=interface_n, metric="RMS retention; relative harm level",
        recommended_use="alternative", status="diagnostic",
    )

    auto_transitions = interface_transition[(interface_transition["condition"] == "auto-FGSM") & np.isclose(interface_transition["epsilon"], PRIMARY_EPSILON)].copy()
    for fig_id, metric, ylabel, caption_metric in (
        ("fig_c3_retention_distribution", "retention_rms", "Rétention RMS", "rétention RMS"),
        ("fig_c3_harm_distribution", "harm_change_normalized", r"$\Delta\widetilde{H}$", "changement de dommage normalisé"),
        ("fig_c3_alignment_distribution", "alignment_change", r"$\Delta C$", "changement d'alignement nuisible"),
    ):
        method = _interface_method(
            fig_id, f"Distribution échantillon-par-échantillon de {caption_metric}.",
            INTERFACE_KEYS, alignment, cohort_summary, "interface",
            "FGSM blanche propre à chaque cible, tenseurs non quantifiés",
            f"Valeur par exemple de {metric}, sans moyenne préalable; boîte = quartiles et médiane, moustaches standard, valeurs extrêmes non dessinées.",
            [
                "La figure mesure la distribution appariée, et non seulement une moyenne architecturale.",
                "La largeur des distributions montre une hétérogénéité importante entre exemples.",
                "Cette hétérogénéité est compatible avec une transmission dépendante de la direction et du point d'entrée.",
                "La classe, la marge propre et le degré de saturation peuvent expliquer une partie de la dispersion.",
                "La distribution ne démontre pas un sous-espace fixe commun à tous les exemples.",
            ],
            "Elle soutient que les moyennes d'architecture masquent des variations individuelles.",
            "Elle ne soutient pas une loi déterministe de l'interface.",
            "Les boîtes regroupent deux graines; les valeurs par graine restent disponibles dans le CSV.",
        )
        render = lambda d, s, m=metric, y=ylabel: plot_distribution(d, s, m, y, reference=1 if m == "retention_rms" else 0)
        add(
            fig_id, auto_transitions, render,
            f"Distribution du {caption_metric} à l'interface classique–quantique sous FGSM blanche à $\\varepsilon=0{{,}}10$. Les valeurs sont appariées par exemple; les extrêmes ne sont pas dessinés mais restent dans le CSV.",
            method, concept=f"distribution de {caption_metric}", models="HVQL; HPVQL (5); HVQM; HQE",
            attack="auto-FGSM", epsilon="0.10", cohort_size=interface_n, metric=metric,
            recommended_use="annexe", status="validé",
        )

    schematic_data = pd.DataFrame([
        {"kind": "selective_schematic", "element": "latent_observable", "x": 2.2, "y": 0.75, "style": "solid"},
        {"kind": "selective_schematic", "element": "latent_non_observable", "x": 1.1, "y": 1.35, "style": "dashed"},
        {"kind": "selective_schematic", "element": "unitary", "x": 3.875, "y": 2.0, "style": "box"},
        {"kind": "selective_schematic", "element": "measurement", "x": 5.925, "y": 2.0, "style": "box"},
        {"kind": "selective_schematic", "element": "measured_displacement", "x": 8.15, "y": 2.55, "style": "arrow"},
        {"kind": "selective_schematic", "element": "harmful_gradient", "x": 8.15, "y": 3.2, "style": "arrow"},
    ])
    schematic_method = _companion_text(
        "fig_c4_selective_projection", "Schéma conceptuel de sélectivité observable.",
        "Schéma générique, informé par les définitions et non ajusté à une architecture unique.",
        "Aucun point de contrôle numérique; le schéma ne remplace pas les figures empiriques C1–C3.",
        "Sans cohorte : schéma conceptuel.", "Aucune", "sans objet",
        "Déplacement latent, évolution unitaire, ensemble restreint d'observables, déplacement mesuré et direction locale nuisible.",
        r"Décomposition conceptuelle Δz=Δz_obs+Δz_non obs; la distance d'état est conservée par U_phi, tandis que M_Z restreint l'information accessible.",
        "Aucune échelle numérique; les longueurs servent uniquement à distinguer les composantes.",
        "Sans agrégation.", "Sans intervalle d'incertitude.", "Sans objet.",
        [
            "Le schéma représente les objets mathématiques mesurés dans C1–C3.",
            "Une composante mesurée plus petite peut être orientée vers la direction nuisible.",
            "Cette géométrie est compatible avec une sélection par observables après évolution unitaire.",
            "Un goulot classique ou une projection non quantique pourrait produire une structure analogue.",
            "Le dessin n'est pas une projection linéaire littérale ni une preuve que l'unitaire débruite.",
        ],
        "Il soutient une formulation précise séparant évolution unitaire et accès par mesure.",
        "Il ne soutient aucune valeur empirique ni spécificité quantique causale.",
        "La décomposition observable/non observable dépend du jeu d'observables et du readout aval.",
    )
    add(
        "fig_c4_selective_projection", schematic_data, plot_selective_projection_schematic,
        "Schéma conceptuel de la sélectivité de l'interface. L'évolution unitaire conserve l'information de l'état, tandis que les observables accessibles au classifieur peuvent transmettre un déplacement mesuré plus faible mais mieux aligné avec la direction locale nuisible. Le dessin n'est pas une projection linéaire littérale.",
        schematic_method, concept="schéma de projection sélective", models="générique",
        attack="aucune", epsilon="sans objet", cohort_size=0, metric="coordonnées conceptuelles",
        recommended_use="corps principal", status="conceptuel",
    )

    _, _, direction_n = _cohort_text(cohort_summary, "direction")
    heat_specs = (
        ("fig_d1_direction_retention", "retention_rms", "Rétention RMS", "rétention RMS"),
        ("fig_d2_direction_harm", "harm_change_normalized", r"$\Delta\widetilde{H}$", "changement de dommage normalisé"),
        ("fig_d3_direction_alignment", "alignment_change", r"$\Delta C$", "changement d'alignement nuisible"),
    )
    for fig_id, metric, label, concept_label in heat_specs:
        data = direction_heatmap_data(direction_transition, metric)
        method = _interface_method(
            fig_id, f"Transmission directionnelle : {concept_label} par source et cible.",
            INTERFACE_KEYS, alignment, cohort_summary, "direction",
            "FGSM transférée depuis CNN, deux MLP, HQuanv, HVQL et VQM; auto-FGSM; contrôle aléatoire apparié en L2",
            f"Chaque cellule est la médiane des médianes de graines de {metric} pour une paire source–cible.",
            [
                "La matrice mesure la compatibilité interne d'une direction source avec chaque interface cible.",
                "Les cellules varient selon la source et la cible plutôt que selon la cible seule.",
                "Cette structure est compatible avec une transmission sélective dépendante de la direction.",
                "La force initiale de l'attaque, les marges et la similarité des encodeurs restent des explications alternatives.",
                "La matrice ne permet pas d'attribuer causalement l'ASR de transfert à l'interface.",
            ],
            "Elle soutient l'existence d'une compatibilité source–interface hétérogène.",
            "Elle ne soutient pas une suppression uniforme par HQE ni une protection garantie par segmentation.",
            "Le contrôle aléatoire est apparié en L2 après clipping; il n'est pas apparié en structure spatiale ni en marge.",
        )
        add(
            fig_id, data, lambda d, s, m=metric, l=label: plot_heatmap(d, s, m, l),
            f"{concept_label.capitalize()} entre latent préquantique et mesures selon la source de perturbation et la cible hybride, à $\\varepsilon=0{{,}}10$. Chaque cellule est une médiane hiérarchique; les valeurs par graine figurent dans le CSV.",
            method, concept=f"transmission directionnelle — {concept_label}", models="HVQL; HPVQL (5); HVQM; HQE",
            attack="auto, transferts et aléatoire L2-apparié", epsilon="0.10", cohort_size=direction_n, metric=metric,
            recommended_use="corps principal" if fig_id != "fig_d3_direction_alignment" else "annexe",
            status="validé",
        )

    asr_data, asr_stats = transmission_asr_data(direction_transition, attack_aggregates)
    stats_dir = output_dir / "fig_d5_transmission_asr"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "fig_d5_transmission_asr_stats.json").write_text(json.dumps(asr_stats, indent=2), encoding="utf-8")
    asr_method = _interface_method(
        "fig_d5_transmission_asr", "Association entre dommage transmis à la mesure et ASR entrant exact.",
        INTERFACE_KEYS, alignment, cohort_summary, "direction",
        "FGSM transférée depuis six sources; ASR exact lu dans les cubes/tables d'attaque complets",
        "Pour chaque graine et paire source–cible externe : x = médiane du dommage normalisé à la mesure sur la cohorte tracée; y = ASR entrant exact sur 10 000 images. Corrélation de Spearman avec bootstrap stratifié par graine.",
        [
            "La figure mesure une association entre une quantité interne diagnostique et le succès de transfert exact.",
            f"La corrélation observée est rho_s={asr_stats['rho']:.2f}, IC bootstrap [{asr_stats['ci_low']:.2f}; {asr_stats['ci_high']:.2f}].",
            "Une association positive est compatible avec une compatibilité source–interface contribuant à l'exposition entrante.",
            "La puissance de la source, la marge de la cible et les encodeurs partagés peuvent confondre cette association.",
            "La corrélation ne permet pas d'établir une causalité ni une médiation complète par l'interface.",
        ],
        "Elle peut soutenir une association descriptive si l'IC n'est pas entièrement non informatif.",
        "Elle ne soutient aucune causalité et ne compare pas des cohortes identiques entre le trace interne et l'ASR complet.",
        "Le prédicteur est estimé sur une cohorte commune réduite, tandis que l'ASR est calculé sur l'ensemble complet.",
    )
    add(
        "fig_d5_transmission_asr", asr_data, lambda d, s: plot_asr_scatter(d, s, asr_stats),
        f"Association entre dommage normalisé transmis à la mesure et ASR entrant exact pour des attaques transférées à $\\varepsilon=0{{,}}10$. Corrélation de Spearman $\\rho_s={asr_stats['rho']:.2f}$, IC bootstrap [{asr_stats['ci_low']:.2f}; {asr_stats['ci_high']:.2f}]. L'association ne constitue pas une preuve causale.",
        asr_method, concept="transmission interne et ASR entrant", models="quatre cibles × six sources × deux graines",
        attack="FGSM transférée", epsilon="0.10", cohort_size=direction_n, metric="measurement harm; incoming ASR",
        recommended_use="annexe", status="association descriptive",
    )

    segmentation_summary = interface_summary_data(
        segmentation_transition, keys=("hvql", "hpvql5", "hpvql10"), condition="auto-FGSM"
    )
    _, _, segmentation_n = _cohort_text(cohort_summary, "segmentation")
    seg_retention = _interface_dot(segmentation_summary, "retention_rms")
    seg_retention["metric"] = "retention_rms"
    seg_method = _interface_method(
        "fig_e1_segmentation_retention", "Comparaison descriptive du couplage global et du traitement partitionné.",
        ("hvql", "hpvql5", "hpvql10"), alignment, cohort_summary, "segmentation",
        "FGSM blanche propre à chaque modèle",
        "Rétention RMS médiane à l'interface, avec IC bootstrap hiérarchique.",
        [
            "La figure mesure la rétention pour HVQL et deux nombres de blocs HPVQL.",
            "Des performances proches peuvent coexister avec des rétentions internes différentes ou proches.",
            "Le motif est compatible avec l'absence de nécessité d'un couplage global complet.",
            "Le nombre de paramètres, la profondeur effective et l'optimisation ne sont pas parfaitement contrôlés.",
            "La figure ne permet pas d'établir que la segmentation protège.",
        ],
        "Elle soutient que le comportement faible-régime n'exige pas nécessairement un circuit global unique.",
        "Elle ne soutient aucune causalité protectrice de la segmentation.",
        "Comparaison de checkpoints entraînés séparément; dimensions et paramètres sont consignés mais non appariés expérimentalement.",
    )
    add(
        "fig_e1_segmentation_retention", seg_retention,
        lambda d, s: plot_dot(d, s, "Rétention RMS"),
        "Rétention RMS à l'interface pour HVQL et HPVQL à cinq et dix blocs, sous FGSM blanche à $\\varepsilon=0{,}10$. La comparaison est descriptive car les configurations ne constituent pas une ablation parfaitement contrôlée.",
        seg_method, concept="couplage global versus partitionné", models="HVQL; HPVQL (5); HPVQL (10)",
        attack="auto-FGSM", epsilon="0.10", cohort_size=segmentation_n, metric="retention_rms",
        recommended_use="annexe", status="confondue mais informative",
    )

    outcomes_seg = attack_outcome_data(attack_aggregates, ("hvql", "hpvql5", "hpvql10"))
    robust_seg = outcomes_seg[outcomes_seg["metric"] == "robust_accuracy"].copy()
    seg_acc_method = _full_attack_method(
        "fig_e1_segmentation_robust_accuracy",
        "Performance faible-régime pour le couplage global et le traitement partitionné.",
        ("hvql", "hpvql5", "hpvql10"), alignment,
        "Exactitude robuste exacte à ε=0,10, recomputée depuis les logits adversariaux sauvegardés.",
        [
            "La figure mesure l'exactitude robuste blanche sur l'ensemble de test complet.",
            "HVQL et les variantes HPVQL conservent toutes une performance faible-régime élevée.",
            "Ce résultat indique qu'un couplage quantique global unique n'est pas nécessaire au profil observé.",
            "Le nombre de paramètres, la profondeur effective et l'optimisation diffèrent entre modèles.",
            "La figure ne permet pas d'établir que la segmentation protège.",
        ],
        "Elle soutient la non-nécessité empirique d'un couplage global complet dans ces configurations.",
        "Elle ne soutient pas un effet causal protecteur du nombre de blocs.",
        "Deux graines et configurations entraînées séparément; comparaison non contrôlée.",
    )
    add(
        "fig_e1_segmentation_robust_accuracy", robust_seg,
        lambda d, s: plot_dot(d, s, "Exactitude robuste (ε = 0,10)", percent=True),
        "Exactitude robuste blanche à $\\varepsilon=0{,}10$ pour HVQL et HPVQL à cinq et dix blocs. Les barres couvrent les deux graines; aucune causalité de segmentation n'est inférée.",
        seg_acc_method, concept="performance faible-régime et segmentation", models="HVQL; HPVQL (5); HPVQL (10)",
        attack="auto-FGSM exacte", epsilon="0.10", cohort_size=10000, metric="robust_accuracy",
        recommended_use="corps principal ou annexe", status="métrique exacte, comparaison confondue",
    )

    responsibility_keys = INTERFACE_KEYS
    outcomes_resp = attack_outcome_data(attack_aggregates, responsibility_keys)
    robust_resp = outcomes_resp[outcomes_resp["metric"] == "robust_accuracy"].copy()
    resp_method = _full_attack_method(
        "fig_f1_responsibility_robust_accuracy", "Robustesse faible-régime selon des architectures à responsabilité quantique différente.",
        responsibility_keys, alignment,
        "Exactitude robuste sur 10 000 images, séparée de la propagation interne diagnostique.",
        [
            "La figure mesure l'exactitude robuste blanche exacte à ε=0,10.",
            "L'ordre des architectures n'est pas monotone avec une responsabilité quantique qualitative plus grande.",
            "Le motif est compatible avec un rôle important de l'encodeur et de la tête classique.",
            "La capacité, les marges et la qualité d'optimisation restent des confondeurs majeurs.",
            "La figure ne quantifie pas causalement la responsabilité quantique.",
        ],
        "Elle soutient l'absence de relation monotone simple entre responsabilité quantique et robustesse.",
        "Elle ne soutient pas qu'une faible ou forte responsabilité quantique cause le résultat.",
        "Les architectures diffèrent simultanément par encodeur, compression, paramètres et readout.",
    )
    add(
        "fig_f1_responsibility_robust_accuracy", robust_resp,
        lambda d, s: plot_dot(d, s, "Exactitude robuste (ε = 0,10)", percent=True),
        "Exactitude robuste blanche à $\\varepsilon=0{,}10$ pour HVQL, HPVQL, HVQM et HQE. Les architectures diffèrent également par capacité et encodeur; la figure teste seulement l'absence de monotonie descriptive.",
        resp_method, concept="responsabilité quantique et robustesse", models="HVQL; HPVQL (5); HVQM; HQE",
        attack="auto-FGSM exacte", epsilon="0.10", cohort_size=10000, metric="robust_accuracy",
        recommended_use="corps principal", status="métrique exacte, mécanisme confondu",
    )

    architecture_seed0 = architecture[(architecture["model_key"].isin(responsibility_keys)) & (architecture["seed"] == 0)].copy()
    for fig_id, metric, ylabel in (
        ("fig_f1_classical_parameters", "classical_parameters", "Paramètres classiques"),
        ("fig_f1_measured_dimension", "measured_output_dimension", "Dimension mesurée"),
    ):
        data = pd.DataFrame({
            "kind": "dot", "model_key": architecture_seed0["model_key"], "model": architecture_seed0["model"],
            "metric": metric, "value": architecture_seed0[metric], "lower": architecture_seed0[metric],
            "upper": architecture_seed0[metric], "n_samples": 1, "n_seeds": 2,
        })
        method = _architecture_method(
            fig_id, f"Confondeur architectural : {ylabel.lower()}.", responsibility_keys, alignment,
            f"{ylabel} dérivé directement du modèle exécutable.",
            [
                f"La figure mesure exactement {ylabel.lower()} pour chaque architecture.",
                "Les architectures diffèrent fortement sur cette quantité structurelle.",
                "Cette hétérogénéité est compatible avec un confondement de la comparaison de responsabilité quantique.",
                "Les paramètres et la dimension mesurée ne résument pas seuls la capacité effective.",
                "La figure ne mesure pas la robustesse et ne lui attribue aucun effet causal.",
            ],
            "Elle documente un confondeur mesuré de la comparaison inter-architectures.",
            "Elle ne soutient aucune relation monotone avec l'exactitude robuste.",
            "Une seule configuration par famille; la capacité fonctionnelle dépend aussi de l'optimisation et des activations.",
        )
        add(
            fig_id, data, lambda d, s, y=ylabel, m=metric: plot_dot(d, s, y, log_scale="parameters" in m),
            f"{ylabel} des quatre hybrides tardifs utilisés dans la comparaison de responsabilité décisionnelle. Cette quantité documente un confondeur architectural et ne constitue pas une mesure de robustesse.",
            method, concept=f"confondeur : {ylabel.lower()}", models="HVQL; HPVQL (5); HVQM; HQE",
            attack="aucune", epsilon="sans objet", cohort_size=0, metric=metric,
            recommended_use="annexe", status="descriptif",
        )

    harm_dot = _interface_dot(interface_summary, "harm_change_normalized")
    harm_dot["metric"] = "harm_change_normalized"
    harm_method = _interface_method(
        "fig_f1_interface_harm_concentration",
        "Concentration du dommage à l'interface selon la responsabilité décisionnelle.",
        responsibility_keys, alignment, cohort_summary, "interface",
        "FGSM blanche propre à chaque cible, tenseurs non quantifiés",
        "Changement de dommage normalisé entre latent préquantique et mesures, agrégé hiérarchiquement.",
        [
            "La figure mesure le changement de dommage projeté à l'interface.",
            "L'ordre n'est pas monotone avec une responsabilité quantique qualitative plus grande.",
            "Le motif est compatible avec une interaction entre encodeur, interface et tête classique.",
            "La normalisation HQE est instable et la capacité diffère fortement entre architectures.",
            "La figure ne quantifie pas causalement la part de décision portée par le circuit.",
        ],
        "Elle soutient l'absence de monotonie simple entre dommage concentré et responsabilité quantique qualitative.",
        "Elle ne soutient pas une causalité propre au quantum ni une comparaison stable de HQE.",
        "HQE ne couvre que 53,5 % des exemples normalisables à ε=0,10 et varie fortement entre graines.",
    )
    add(
        "fig_f1_interface_harm_concentration", harm_dot,
        lambda d, s: plot_dot(d, s, r"Concentration $\Delta\widetilde{H}$"),
        "Changement de dommage normalisé entre latent préquantique et mesures pour quatre hybrides tardifs. Une valeur positive indique une concentration locale du dommage malgré la transformation de représentation.",
        harm_method, concept="concentration du dommage et responsabilité", models="HVQL; HPVQL (5); HVQM; HQE",
        attack="auto-FGSM", epsilon="0.10", cohort_size=interface_n, metric="harm_change_normalized",
        recommended_use="annexe proche", status="validé",
    )

    _, _, residual_n = _cohort_text(cohort_summary, "residual")
    residual_l2 = residual_summary_data(residual, "delta_l2_contribution")
    residual_rms = residual_summary_data(residual, "delta_rms_contribution")
    residual_harm = residual_summary_data(residual, "projected_margin_harm")
    q_l2 = float(residual_l2.loc[residual_l2["branch"] == "quantum_branch", "value"].iloc[0])
    r_l2 = float(residual_l2.loc[residual_l2["branch"] == "residual_branch", "value"].iloc[0])
    branch_ratio = r_l2 / max(q_l2, 1e-12)
    residual_base = _interface_method(
        "fig_g1_residual_l2", "Propagation par les branches du HVQL résiduel.",
        ("reshvql",), alignment, cohort_summary, "residual",
        "FGSM blanche propre au HVQL résiduel",
        "La branche classique est multipliée par l'échelle résiduelle entraînée avant de calculer sa contribution de déplacement; H de branche utilise le gradient exact de la sortie de branche, qui inclut l'échelle dans la dérivée.",
        [
            "La figure mesure séparément la branche quantique, le contournement classique et leur fusion.",
            f"Le rapport médian L2 contournement/quantique recomputé vaut {branch_ratio:.2f}.",
            "Un contournement dominant est compatible avec le transport d'un déplacement autour de l'interface mesurée.",
            "La projection résiduelle dense, son échelle et sa sensibilité aval peuvent expliquer le rapport sans propriété quantique spécifique.",
            "La comparaison ne prouve pas que le goulot quantique seul cause la robustesse du modèle standard.",
        ],
        "Elle soutient une voie fonctionnelle de contournement portant une part mesurable du déplacement adversarial.",
        "Elle ne soutient pas une attribution causale au caractère quantique de l'autre branche.",
        "Diagnostic d'inférence sur un modèle entraîné avec résidu; aucune ablation réentraînée n'est disponible.",
    )
    for fig_id, data, ylabel, metric, use in (
        ("fig_g1_residual_l2", residual_l2, r"Déplacement L2 de contribution", "delta_l2_contribution", "annexe"),
        ("fig_g1_residual_rms", residual_rms, r"Déplacement RMS par composante", "delta_rms_contribution", "corps principal"),
        ("fig_g2_residual_harm", residual_harm, r"Dommage projeté $H$", "projected_margin_harm", "annexe proche"),
    ):
        method = residual_base.replace("fig_g1_residual_l2", fig_id)
        add(
            fig_id, data, lambda d, s, y=ylabel: plot_bar(d, s, y),
            f"{ylabel} pour la branche quantique, le contournement classique mis à l'échelle et la représentation fusionnée du HVQL résiduel, sous FGSM blanche à $\\varepsilon=0{{,}}10$. Les barres d'erreur sont des IC bootstrap hiérarchiques à 95\,\%.",
            method, concept=f"branches résiduelles — {metric}", models="HVQL résiduel",
            attack="auto-FGSM", epsilon="0.10", cohort_size=residual_n, metric=metric,
            recommended_use=use, status="validé",
        )

    routing_method = residual_base.replace("fig_g1_residual_l2", "fig_g4_residual_routing").replace(
        "La figure mesure séparément", "Le schéma encode par l'épaisseur des flèches les médianes RMS mesurées séparément"
    )
    add(
        "fig_g4_residual_routing", residual_rms, plot_residual_schematic,
        "Schéma de routage du HVQL résiduel. L'épaisseur des flèches est proportionnelle au déplacement RMS médian recomputé dans chaque branche; aucune statistique de performance n'est inscrite dans le dessin.",
        routing_method, concept="schéma du contournement résiduel", models="HVQL résiduel",
        attack="auto-FGSM", epsilon="0.10", cohort_size=residual_n, metric="arrow width from median branch RMS",
        recommended_use="corps principal ou annexe", status="schéma informé par les données",
    )

    manifest = pd.DataFrame(manifests)
    manifest.to_csv(output_dir / "figure_manifest.csv", index=False)
    return manifest

"""Mechanistic clean/adversarial tracing for the MNIST attack benchmark.

The module is intentionally library-only.  The companion notebook
``mechanistic_propagation.ipynb`` owns discovery, model selection, execution,
and figure export.

The saved attack audit images are used as the perturbation source of truth.
This preserves the original source-model FGSM directions and avoids rerunning
attacks.  Full-dataset outcome metrics are read from the aggregate parquet
chunks; intermediate activations are measured on the fixed audit cohort.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


DEFAULT_MODEL_ROOT = Path("logs/ByModel")
DEFAULT_ATTACK_DIR = Path("logs/ATTACKS/attack_experiment")
DEFAULT_OUTPUT_DIR = Path("outputs/mechanistic_propagation")


@dataclass(frozen=True)
class BlockCheckpoint:
    """A meaningful PyTorch module output in the model's forward path."""

    module_path: str
    label: str
    transform: str
    scope: str
    order: float
    path: str = "main"
    note: str = ""


@dataclass(frozen=True)
class ModelTraceSpec:
    """Checkpoint, attack-name, and hook specification for one trained model."""

    key: str
    display_name: str
    family: str
    architecture: str
    attack_name: str
    model_dir: str
    blocks: tuple[BlockCheckpoint, ...]
    amplitude_source: str | None = None
    amplitude_blocks: int = 1
    artifact_note: str = ""


REPRESENTATIVE_SOURCES: Mapping[str, str] = {
    "CNN": "Classical/convnet/conv_channels=[32,64],fc_sizes=[128]",
    "MLP": "Classical/mlp/hidden_sizes=[128,64]",
    "HQuanv": "Hybrid/hquanv",
    # Historical attack folder labels are swapped relative to the executable
    # config names: raw ``hvqm/ansatz=*`` loads the rich HVQL layer, while raw
    # ``hvql/ansatz=*`` loads the direct VQL classifier (HVQM semantics).
    "HVQL": "Hybrid/hvqm/ansatz=strong,qdepth=50",
    "VQM50": "Quantum/VQM50-Strong",
}


def default_model_specs() -> dict[str, ModelTraceSpec]:
    """Return the attacked/near-attacked variants used by the notebook.

    Paths under ``ByBenchmark`` are used when they reproduce the saved attack
    baselines more closely than later copies under ``ByModel``.
    """

    conv_encoder = (
        BlockCheckpoint("encoder.conv1", "conv 1", "5x5 convolution", "local", 1.0),
        BlockCheckpoint("encoder.maxpool1", "pool 1", "ReLU + 2x2 max-pool", "local", 2.0),
        BlockCheckpoint("encoder.conv2", "conv 2", "5x5 convolution", "local/hierarchical", 3.0),
        BlockCheckpoint("encoder", "encoder output", "pool + flatten", "hierarchical", 4.0),
    )
    late_hybrid = (
        BlockCheckpoint("classifier.fc1", "interface dense", "dense projection", "global/compressive", 5.0),
        BlockCheckpoint("classifier.relu", "pre-quantum latent", "batch norm + ReLU", "global latent", 6.0),
        BlockCheckpoint("classifier.qlayer", "measured quantum vector", "VQC + Pauli-Z measurements", "global latent", 8.0),
        BlockCheckpoint("classifier.fc2", "classical head", "dense readout", "global", 9.0),
    )
    specs = {
        "cnn": ModelTraceSpec(
            "cnn", "CNN", "Classical", "CNN",
            REPRESENTATIVE_SOURCES["CNN"],
            "logs/ByModel/Classical/convnet/conv_channels=[32,64],fc_sizes=[128]",
            (
                BlockCheckpoint("features.0", "conv 1", "3x3 convolution", "local", 1.0),
                BlockCheckpoint("features.2", "pool 1", "ReLU + 2x2 max-pool", "local", 2.0),
                BlockCheckpoint("features.3", "conv 2", "3x3 convolution", "local/hierarchical", 3.0),
                BlockCheckpoint("features.5", "pool 2", "ReLU + 2x2 max-pool", "hierarchical", 4.0),
                BlockCheckpoint("classifier.0", "dense hidden", "dense projection", "global", 5.0),
                BlockCheckpoint("classifier.1", "dense ReLU", "ReLU", "global", 6.0),
            ),
        ),
        "resnet18": ModelTraceSpec(
            "resnet18", "ResNet18", "Classical", "ResNet18",
            "Classical/resnet18", "logs/ByModel/Classical/resnet18",
            (
                BlockCheckpoint("conv1", "stem convolution", "3x3 convolution", "local", 1.0),
                BlockCheckpoint("relu", "stem activation", "batch norm + ReLU", "local", 2.0),
                BlockCheckpoint("layer1", "residual stage 1", "two residual blocks", "hierarchical/residual", 3.0),
                BlockCheckpoint("layer2", "residual stage 2", "downsample + residual blocks", "hierarchical/residual", 4.0),
                BlockCheckpoint("layer3", "residual stage 3", "downsample + residual blocks", "hierarchical/residual", 5.0),
                BlockCheckpoint("layer4", "residual stage 4", "downsample + residual blocks", "hierarchical/residual", 6.0),
                BlockCheckpoint("avgpool", "global average pool", "spatial pooling", "global", 7.0),
                BlockCheckpoint("linear", "linear head", "dense readout", "global", 8.0),
            ),
        ),
        "mlp": ModelTraceSpec(
            "mlp", "MLP (128-64)", "Classical", "MLP",
            REPRESENTATIVE_SOURCES["MLP"], "logs/ByModel/Classical/mlp/hidden_sizes=[128,64]",
            (
                BlockCheckpoint("hidden_layers.0", "dense 784->128", "dense, followed by ReLU", "global/raw-pixel", 1.0),
                BlockCheckpoint("hidden_layers.1", "dense 128->64", "dense, followed by ReLU", "global", 2.0),
                BlockCheckpoint("output_layer", "linear head", "dense readout", "global", 3.0),
            ),
            artifact_note=(
                "The higher-capacity folder named hidden_sizes=[1024, 512, 64, 64] contains a LeNet "
                "checkpoint and is excluded from MLP mechanistic claims."
            ),
        ),
        "mlpmixer": ModelTraceSpec(
            "mlpmixer", "MLPMixer", "Classical", "MLPMixer",
            "Classical/mlpmixer", "logs/ByModel/Classical/mlpmixer",
            (
                BlockCheckpoint("patchemb.proj", "patch projection", "4x4 stride-4 convolution", "local patches", 1.0),
                BlockCheckpoint("patchemb", "patch tokens", "reshape to 49 tokens", "local patches", 2.0),
                BlockCheckpoint("blocks.0", "mixer block 1", "global token mixing + channel mixing + residuals", "global tokens", 3.0),
                BlockCheckpoint("blocks.1", "mixer block 2", "global token mixing + channel mixing + residuals", "global tokens", 4.0),
                BlockCheckpoint("blocks.2", "mixer block 3", "global token mixing + channel mixing + residuals", "global tokens", 5.0),
                BlockCheckpoint("blocks.3", "mixer block 4", "global token mixing + channel mixing + residuals", "global tokens", 6.0),
                BlockCheckpoint("norm", "token normalization", "layer norm", "global tokens", 7.0),
                BlockCheckpoint("head", "linear head", "mean pool + dense readout", "global", 8.0),
            ),
        ),
        "hvqm": ModelTraceSpec(
            "hvqm", "HVQM (raw artifact: HVQL, Strong d=50)", "Hybrid", "HVQM",
            "Hybrid/hvql/ansatz=strong,qdepth=50",
            "logs/ByBenchmark/HVQL-Ansatzs/ansatz=strong,qdepth=50",
            conv_encoder + (
                BlockCheckpoint("classifier.vqc", "measured quantum logits", "10-qubit VQC + Pauli-Z measurements", "global latent", 6.0),
            ),
            amplitude_source="encoder output",
            artifact_note=(
                "Historical attack path says HVQL, but the checkpoint is the current hvqm configuration: "
                "small 8/16 convolutional encoder, direct VQL measurements, and no learned post-quantum head."
            ),
        ),
        "hpvql": ModelTraceSpec(
            "hpvql", "HPVQL (5 blocks, d=50)", "Hybrid", "HPVQL",
            "Hybrid/hpvql", "logs/ByBenchmark/Baseline-Models/Hybrid/hpvql",
            conv_encoder + late_hybrid,
            amplitude_source="pre-quantum latent", amplitude_blocks=5,
            artifact_note=(
                "The available checkpoint matches all audit predictions but not the stored attack logits "
                "bit-for-bit; alignment is measured and reported."
            ),
        ),
        "hvql": ModelTraceSpec(
            "hvql", "HVQL (raw artifact: HVQM, Strong d=50)", "Hybrid", "HVQL",
            REPRESENTATIVE_SOURCES["HVQL"],
            "logs/ByBenchmark/HVQM-Benchmarks/ansatz=strong,qdepth=50",
            conv_encoder + late_hybrid,
            amplitude_source="pre-quantum latent",
            artifact_note=(
                "Historical attack path says HVQM, but the checkpoint is the current hvql configuration: "
                "16/32 encoder, 1568->1024 interface, VQC, and learned 10->10 classical head."
            ),
        ),
        "hqe": ModelTraceSpec(
            "hqe", "HQE", "Hybrid", "HQE", "Hybrid/hqe",
            "logs/ByBenchmark/Baseline-Models/Hybrid/hqe",
            (
                BlockCheckpoint("encoder.net.0", "dense 784->4096", "dense projection", "global/raw-pixel", 1.0),
                BlockCheckpoint("encoder.net.1", "ReLU 4096", "ReLU", "global", 2.0),
                BlockCheckpoint("encoder.net.2", "dense 4096->2048", "dense compression", "global/compressive", 3.0),
                BlockCheckpoint("encoder.net.3", "ReLU 2048", "ReLU", "global", 4.0),
                BlockCheckpoint("encoder.net.4", "dense 2048->1024", "dense compression", "global/compressive", 5.0),
                BlockCheckpoint("encoder.net.5", "encoder latent 1024", "ReLU", "global latent", 6.0),
                BlockCheckpoint("classifier.fc1", "interface dense", "dense 1024->1024", "global latent", 7.0),
                BlockCheckpoint("classifier.relu", "pre-quantum latent", "batch norm + ReLU", "global latent", 8.0),
                BlockCheckpoint("classifier.qlayer", "measured quantum vector", "10-qubit VQC + Pauli-Z measurements", "global latent", 10.0),
                BlockCheckpoint("classifier.fc2", "classical head", "dense 10->10", "global", 11.0),
            ),
            amplitude_source="pre-quantum latent",
        ),
        "hquanv": ModelTraceSpec(
            "hquanv", "HQuanv", "Hybrid", "HQuanv", REPRESENTATIVE_SOURCES["HQuanv"],
            "logs/ByBenchmark/Baseline-Models/Hybrid/hquanv",
            (
                BlockCheckpoint("encoder.quanv", "quanvolution map", "shared 2x2 patch VQC + linear readout", "local patches", 1.0),
                BlockCheckpoint("encoder", "flattened quanvolution", "flatten", "local patches", 2.0),
                BlockCheckpoint("classifier.net.0", "dense hidden", "dense 784->784", "global", 3.0),
                BlockCheckpoint("classifier.net.1", "dense ReLU", "ReLU", "global", 4.0),
                BlockCheckpoint("classifier.net.2", "linear head", "dense 784->10", "global", 5.0),
            ),
        ),
        "drophvql": ModelTraceSpec(
            "drophvql", "Dropout HVQL", "Hybrid", "DropHVQL", "Hybrid/hvql/dropout",
            "logs/ByModel/Hybrid/hvql/dropout",
            conv_encoder + (
                BlockCheckpoint("classifier.fc1", "interface dense", "dense projection", "global/compressive", 5.0),
                BlockCheckpoint("classifier.relu", "pre-dropout latent", "batch norm + ReLU", "global latent", 6.0),
                BlockCheckpoint("classifier.dropout", "inference dropout", "identity at eval", "global latent", 7.0),
                BlockCheckpoint("classifier.vqc", "measured quantum vector", "VQC + Pauli-Z measurements", "global latent", 9.0),
                BlockCheckpoint("classifier.fc_out", "classical head", "dense readout", "global", 10.0),
            ),
            amplitude_source="inference dropout",
        ),
        "reshvql": ModelTraceSpec(
            "reshvql", "Residual HVQL", "Hybrid", "ResHVQL", "Hybrid/hvql/residuals",
            "logs/ByModel/Hybrid/hvql/residuals",
            conv_encoder + (
                BlockCheckpoint("classifier.fc1", "interface dense", "dense projection", "global/compressive", 5.0),
                BlockCheckpoint("classifier.relu", "pre-quantum latent", "batch norm + ReLU", "global latent", 6.0),
                BlockCheckpoint("classifier.vqc", "quantum branch", "VQC measurements", "global latent", 8.0, "quantum"),
                BlockCheckpoint("classifier.residual_proj", "classical bypass", "dense 1568->10", "global residual", 8.1, "residual"),
                BlockCheckpoint("classifier", "residual merge", "quantum + scaled classical bypass", "global residual", 9.0, "merge"),
            ),
            amplitude_source="pre-quantum latent",
        ),
        "vqm50": ModelTraceSpec(
            "vqm50", "VQM Strong d=50", "Quantum", "VQM", "Quantum/VQM50-Strong",
            "logs/ByBenchmark/VQM-Ansatzs/VQM50-Strong",
            (
                BlockCheckpoint("vqc", "measured observables", "10-qubit VQC + Pauli-Z measurements", "global/raw-pixel state", 2.0),
                BlockCheckpoint("post", "post map", "identity (10 observables)", "global", 3.0),
            ),
            amplitude_source="raw input",
        ),
        "vqm100": ModelTraceSpec(
            "vqm100", "VQM Strong d=100", "Quantum", "VQM", "Quantum/VQM100",
            "logs/ByBenchmark/Baseline-Models/Quantum/VQM100",
            (
                BlockCheckpoint("vqc", "measured observables", "10-qubit VQC + Pauli-Z measurements", "global/raw-pixel state", 2.0),
                BlockCheckpoint("post", "post map", "identity (10 observables)", "global", 3.0),
            ),
            amplitude_source="raw input",
        ),
        "vqm200": ModelTraceSpec(
            "vqm200", "VQM Strong d=200", "Quantum", "VQM", "Quantum/VQM200",
            "logs/ByBenchmark/Baseline-Models/Quantum/VQM200",
            (
                BlockCheckpoint("vqc", "measured observables", "10-qubit VQC + Pauli-Z measurements", "global/raw-pixel state", 2.0),
                BlockCheckpoint("post", "post map", "identity (10 observables)", "global", 3.0),
            ),
            amplitude_source="raw input",
        ),
    }
    return specs


def safe_model_name(name: str) -> str:
    """Return the attack pipeline's filesystem-safe model name."""

    return str(name).replace("\\", "__").replace("/", "__").replace(" ", "_").replace(":", "_")


def crawl_model_dirs(root: str | Path = DEFAULT_MODEL_ROOT) -> list[Path]:
    """Discover valid model folders exclusively through ``.model`` anchors."""

    root = Path(root)
    return sorted(path.parent for path in root.rglob(".model"))


def load_attack_aggregates(attack_dir: str | Path = DEFAULT_ATTACK_DIR) -> pd.DataFrame:
    """Load and validate all chunked aggregate attack tables."""

    table_dir = Path(attack_dir) / "tables"
    parquet_files = sorted(table_dir.glob("*.parquet"))
    csv_files = sorted(table_dir.glob("*.csv"))
    if parquet_files:
        frames = [pd.read_parquet(path, engine="fastparquet") for path in tqdm(parquet_files, desc="Attack tables")]
    elif csv_files:
        frames = [pd.read_csv(path) for path in tqdm(csv_files, desc="Attack tables")]
    else:
        raise FileNotFoundError(f"No parquet/csv attack chunks under {table_dir}")
    table = pd.concat(frames, ignore_index=True)
    required = {
        "seed", "source_model", "target_model", "epsilon", "robust_accuracy", "ASR",
        "flip_rate", "mean_confidence_drop_cc", "mean_margin_drop_cc", "n_samples",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Attack table missing columns: {sorted(missing)}")
    duplicate_count = int(table.duplicated(["seed", "source_model", "target_model", "epsilon"]).sum())
    if duplicate_count:
        raise ValueError(f"Attack table contains {duplicate_count} duplicate source-target-epsilon rows")
    return table


def _auc(frame: pd.DataFrame, value: str) -> float:
    frame = frame.sort_values("epsilon")
    return float(np.trapezoid(frame[value].to_numpy(float), frame["epsilon"].to_numpy(float)))


def canonical_architecture(model_name: str) -> str:
    """Map attack artifact paths to architecture labels without hiding variants."""

    name = str(model_name).replace("\\", "/")
    lower = name.lower()
    if lower == "classical/mlp/hidden_sizes=[1024, 512, 64, 64]":
        return "LeNet (mislabeled as MLP)"
    rules = (
        ("classical/convnet", "CNN"), ("classical/lenet", "LeNet"),
        ("classical/mlp/", "MLP"), ("classical/mlpmixer", "MLPMixer"),
        ("classical/resnet18", "ResNet18"), ("hybrid/hpvql", "HPVQL"),
        ("hybrid/hpvqm", "HPVQM"), ("hybrid/hqe", "HQE"),
        ("hybrid/hquanv", "HQuanv"), ("hybrid/hvql/dropout", "DropHVQL"),
        ("hybrid/hvql/residuals", "ResHVQL"), ("hybrid/hvql/", "HVQM"),
        ("hybrid/hvqm/", "HVQL"), ("quantum/vqm", "VQM"),
    )
    for needle, label in rules:
        if lower.startswith(needle):
            return label
    return name.split("/")[-1]


def model_family(model_name: str) -> str:
    return str(model_name).replace("\\", "/").split("/", 1)[0]


def summarize_attack_metrics(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute seed/variant outcomes and architecture-level summaries.

    White-box metrics use source == target. Incoming and outgoing metrics use
    only source != target pairs.  AUC is integrated over the exact 0..1 grid.
    """

    eps_lookup = {0.0: "000", 0.05: "005", 0.10: "010", 0.30: "030"}
    rows: list[dict[str, object]] = []
    models = sorted(set(table["target_model"]) & set(table["source_model"]))
    for seed in sorted(table["seed"].unique()):
        seed_df = table[table["seed"] == seed]
        for model in models:
            white = seed_df[(seed_df["source_model"] == model) & (seed_df["target_model"] == model)]
            if white.empty:
                continue
            row: dict[str, object] = {
                "seed": int(seed), "model": model, "family": model_family(model),
                "architecture": canonical_architecture(model), "robust_accuracy_auc": _auc(white, "robust_accuracy"),
                "whitebox_asr_auc": _auc(white, "ASR"),
            }
            for eps, suffix in eps_lookup.items():
                point = white[np.isclose(white["epsilon"], eps)]
                if len(point) != 1:
                    raise ValueError(f"Expected one white-box row for {model}, seed={seed}, epsilon={eps}")
                row[f"robust_accuracy_{suffix}"] = float(point["robust_accuracy"].iloc[0])
                row[f"whitebox_asr_{suffix}"] = float(point["ASR"].iloc[0])
                row[f"margin_drop_cc_{suffix}"] = float(point["mean_margin_drop_cc"].iloc[0])
                row[f"confidence_drop_cc_{suffix}"] = float(point["mean_confidence_drop_cc"].iloc[0])
            clean = float(row["robust_accuracy_000"])
            for suffix in ("005", "010", "030"):
                row[f"retention_{suffix}"] = float(row[f"robust_accuracy_{suffix}"]) / clean if clean else np.nan

            incoming = seed_df[(seed_df["target_model"] == model) & (seed_df["source_model"] != model)]
            outgoing = seed_df[(seed_df["source_model"] == model) & (seed_df["target_model"] != model)]
            for direction, frame in (("incoming", incoming), ("outgoing", outgoing)):
                by_eps = frame.groupby("epsilon", as_index=False)["ASR"].mean()
                row[f"{direction}_asr_auc"] = _auc(by_eps, "ASR")
                for eps, suffix in eps_lookup.items():
                    point = by_eps[np.isclose(by_eps["epsilon"], eps)]
                    row[f"{direction}_asr_{suffix}"] = float(point["ASR"].iloc[0])
            rows.append(row)
    variant_seed = pd.DataFrame(rows)
    numeric = [column for column in variant_seed.columns if column not in {"seed", "model", "family", "architecture"}]
    variant = (
        variant_seed.groupby(["model", "family", "architecture"], as_index=False)[numeric]
        .agg(["mean", "min", "max"])
    )
    variant.columns = [
        "__".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in variant.columns
    ]
    architecture = (
        variant_seed.groupby(["family", "architecture"], as_index=False)[numeric]
        .agg(["mean", "std", "min", "max"])
    )
    architecture.columns = [
        "__".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in architecture.columns
    ]
    return variant_seed, architecture


def common_correct_audit_positions(
    attack_dir: str | Path,
    specs: Sequence[ModelTraceSpec],
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positions, dataset indices, and labels clean-correct for all specs."""

    seed_dir = Path(attack_dir) / f"seed_{seed}"
    audit_indices = np.load(seed_dir / "audit_indices.npy")
    labels = np.load(seed_dir / "labels.npy")[audit_indices]
    keep = np.ones(len(audit_indices), dtype=bool)
    for spec in specs:
        preds = np.load(seed_dir / "baselines" / safe_model_name(spec.attack_name) / "preds_clean.npy")
        keep &= preds[audit_indices] == labels
    positions = np.flatnonzero(keep)
    return positions, audit_indices[positions], labels[positions]


def _best_checkpoint(model_dir: Path, seed: int) -> Path:
    checkpoint_dir = model_dir / str(seed) / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("*.ckpt"))
    if not candidates:
        final = model_dir / str(seed) / "final.ckpt"
        if not final.exists():
            raise FileNotFoundError(f"No checkpoint under {model_dir / str(seed)}")
        return final

    def score(path: Path) -> float:
        match = re.search(r"loss=(\d+(?:\.\d+)?)", path.name)
        return float(match.group(1)) if match else math.inf

    return min(candidates, key=lambda path: (score(path), path.name))


def load_traced_model(spec: ModelTraceSpec, project_root: str | Path, seed: int = 0):
    """Load one exact best checkpoint through the project's model registry."""

    from models.ModelFactory import ModelFactory

    model_dir = Path(project_root) / spec.model_dir
    model_name = (model_dir / ".model").read_text(encoding="utf-8").strip()
    checkpoint = _best_checkpoint(model_dir, seed)
    model = ModelFactory.load(
        model_name,
        checkpoint=str(checkpoint),
        map_location="cpu",
        log_dir=str(model_dir / str(seed)),
    )
    model.eval()
    return model, checkpoint


def _resolve_module(model, module_path: str):
    modules = dict(model.named_modules())
    if module_path not in modules:
        available = ", ".join(sorted(modules))
        raise KeyError(f"Module {module_path!r} not found. Available: {available}")
    return modules[module_path]


def _tensor_output(output):
    if hasattr(output, "shape"):
        return output
    if isinstance(output, (tuple, list)) and output and hasattr(output[0], "shape"):
        return output[0]
    raise TypeError(f"Hook output is not tensor-like: {type(output)!r}")


def capture_clean_reference(model, spec: ModelTraceSpec, clean_images, labels) -> dict[str, object]:
    """Capture clean activations and their final-margin gradients in one pass."""

    import torch

    raw: dict[str, object] = {}
    handles = []
    for block in spec.blocks:
        module = _resolve_module(model, block.module_path)

        def hook(_module, _inputs, output, key=block.label):
            tensor = _tensor_output(output)
            if tensor.requires_grad:
                tensor.retain_grad()
            raw[key] = tensor

        handles.append(module.register_forward_hook(hook))
    x = clean_images.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    logits.retain_grad()
    row_index = torch.arange(len(labels), device=logits.device)
    competitor_scores = logits.detach().clone()
    competitor_scores[row_index, labels] = -torch.inf
    competitor = competitor_scores.argmax(dim=1)
    margin = logits[row_index, labels] - logits[row_index, competitor]
    margin.sum().backward()

    activations = {"raw input": x.detach().to(torch.float32).cpu()}
    gradients = {"raw input": x.grad.detach().to(torch.float32).cpu()}
    for block in spec.blocks:
        tensor = raw[block.label]
        activations[block.label] = tensor.detach().to(torch.float32).cpu()
        gradients[block.label] = (
            tensor.grad.detach().to(torch.float32).cpu() if tensor.grad is not None else None
        )
    activations["final logits"] = logits.detach().to(torch.float32).cpu()
    gradients["final logits"] = logits.grad.detach().to(torch.float32).cpu()
    for handle in handles:
        handle.remove()
    return {
        "activations": activations,
        "gradients": gradients,
        "competitor": competitor.detach().cpu(),
        "margin": margin.detach().to(torch.float32).cpu(),
        "logits": logits.detach().to(torch.float32).cpu(),
    }


def capture_activations(model, spec: ModelTraceSpec, images) -> dict[str, object]:
    """Capture detached activations for a perturbed batch."""

    import torch

    activations: dict[str, object] = {}
    handles = []
    for block in spec.blocks:
        module = _resolve_module(model, block.module_path)

        def hook(_module, _inputs, output, key=block.label):
            activations[key] = _tensor_output(output).detach().to(torch.float32).cpu().clone()

        handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        logits = model(images).detach().to(torch.float32).cpu()
    for handle in handles:
        handle.remove()
    return {"raw input": images.detach().to(torch.float32).cpu(), **activations, "final logits": logits}


def _pad_power_of_two(values):
    import torch

    width = values.shape[1]
    padded = 1 << int(math.ceil(math.log2(max(1, width))))
    if padded == width:
        return values
    return torch.nn.functional.pad(values, (0, padded - width))


def amplitude_embedding_state(values, n_blocks: int = 1):
    """Return the real normalized amplitude vector(s) used before the unitary."""

    import torch

    flat = values.reshape(values.shape[0], -1).to(torch.float32)
    if n_blocks <= 1:
        padded = _pad_power_of_two(flat)
        return padded / padded.norm(dim=1, keepdim=True).clamp_min(1e-12)
    block_size = int(math.ceil(flat.shape[1] / n_blocks))
    states = []
    for block_index in range(n_blocks):
        chunk = flat[:, block_index * block_size : (block_index + 1) * block_size]
        if chunk.shape[1] < block_size:
            chunk = torch.nn.functional.pad(chunk, (0, block_size - chunk.shape[1]))
        padded = _pad_power_of_two(chunk)
        states.append(padded / padded.norm(dim=1, keepdim=True).clamp_min(1e-12))
    return torch.cat(states, dim=1)


def _true_confidence(logits, labels):
    import torch

    probabilities = torch.softmax(logits, dim=1)
    return probabilities[torch.arange(len(labels)), labels]


def _metric_rows(
    spec: ModelTraceSpec,
    clean_reference: Mapping[str, object],
    perturbed: Mapping[str, object],
    labels,
    condition: str,
    source_model: str,
    epsilon: float,
) -> list[dict[str, object]]:
    import torch

    clean_acts = dict(clean_reference["activations"])
    adv_acts = dict(perturbed)
    gradients = dict(clean_reference["gradients"])
    block_meta = {
        "raw input": BlockCheckpoint("", "raw input", "identity", "raw pixels", 0.0),
        **{block.label: block for block in spec.blocks},
        "final logits": BlockCheckpoint("", "final logits", "trained decision", "decision", 99.0),
    }
    if spec.amplitude_source:
        source = spec.amplitude_source
        clean_state = amplitude_embedding_state(clean_acts[source], spec.amplitude_blocks)
        adv_state = amplitude_embedding_state(adv_acts[source], spec.amplitude_blocks)
        clean_acts["amplitude state"] = clean_state
        adv_acts["amplitude state"] = adv_state
        gradients["amplitude state"] = None
        source_order = block_meta[source].order if source in block_meta else 0.0
        block_meta["amplitude state"] = BlockCheckpoint(
            "", "amplitude state", "normalize + zero-pad amplitude embedding",
            "global quantum state", source_order + 0.5,
            note="Pure-state fidelity is preserved by the subsequent noiseless unitary.",
        )

    input_delta = (adv_acts["raw input"] - clean_acts["raw input"]).reshape(len(labels), -1)
    input_l2 = input_delta.norm(dim=1).clamp_min(1e-12)
    input_rms = input_l2 / math.sqrt(input_delta.shape[1])
    input_linf = input_delta.abs().amax(dim=1).clamp_min(1e-12)
    clean_logits = clean_acts["final logits"]
    adv_logits = adv_acts["final logits"]
    row_index = torch.arange(len(labels))
    clean_other = clean_logits.clone()
    adv_other = adv_logits.clone()
    clean_other[row_index, labels] = -torch.inf
    adv_other[row_index, labels] = -torch.inf
    clean_margin = clean_logits[row_index, labels] - clean_other.amax(dim=1)
    adv_margin = adv_logits[row_index, labels] - adv_other.amax(dim=1)
    clean_conf = _true_confidence(clean_logits, labels)
    adv_conf = _true_confidence(adv_logits, labels)

    rows: list[dict[str, object]] = []
    for layer, clean in clean_acts.items():
        adv = adv_acts[layer]
        clean_flat = clean.reshape(len(labels), -1)
        adv_flat = adv.reshape(len(labels), -1)
        delta = adv_flat - clean_flat
        delta_l2 = delta.norm(dim=1)
        delta_rms = delta_l2 / math.sqrt(delta.shape[1])
        delta_linf = delta.abs().amax(dim=1)
        clean_l2 = clean_flat.norm(dim=1)
        cosine = torch.nn.functional.cosine_similarity(clean_flat, adv_flat, dim=1, eps=1e-12)
        grad = gradients.get(layer)
        if grad is not None:
            grad_flat = grad.reshape(len(labels), -1)
            dot = (grad_flat * delta).sum(dim=1)
            alignment = torch.nn.functional.cosine_similarity(delta, grad_flat, dim=1, eps=1e-12)
            predicted_drop = -dot
            harm_per_l2 = predicted_drop / delta_l2.clamp_min(1e-12)
        else:
            alignment = torch.full((len(labels),), torch.nan)
            predicted_drop = torch.full((len(labels),), torch.nan)
            harm_per_l2 = torch.full((len(labels),), torch.nan)

        fidelity = torch.full((len(labels),), torch.nan)
        trace_distance = torch.full((len(labels),), torch.nan)
        if layer == "amplitude state":
            if spec.amplitude_blocks <= 1:
                overlap = (clean_flat * adv_flat).sum(dim=1).abs().clamp(max=1.0)
                fidelity = overlap.square()
            else:
                block_width = clean_flat.shape[1] // spec.amplitude_blocks
                parts = []
                for block_index in range(spec.amplitude_blocks):
                    c = clean_flat[:, block_index * block_width : (block_index + 1) * block_width]
                    a = adv_flat[:, block_index * block_width : (block_index + 1) * block_width]
                    parts.append((c * a).sum(dim=1).abs().clamp(max=1.0).square())
                fidelity = torch.stack(parts, dim=1).mean(dim=1)
            trace_distance = torch.sqrt((1.0 - fidelity).clamp_min(0.0))

        meta = block_meta[layer]
        for sample_index in range(len(labels)):
            rows.append({
                "model_key": spec.key, "model": spec.display_name, "family": spec.family,
                "architecture": spec.architecture, "attack_target": spec.attack_name,
                "condition": condition, "source_model": source_model, "epsilon": float(epsilon),
                "sample": int(sample_index), "label": int(labels[sample_index]),
                "layer": layer, "layer_order": float(meta.order), "path": meta.path,
                "transform": meta.transform, "scope": meta.scope, "shape": "x".join(map(str, clean.shape[1:])),
                "width": int(clean_flat.shape[1]),
                "delta_l2": float(delta_l2[sample_index]),
                "delta_rms": float(delta_rms[sample_index]),
                "delta_linf": float(delta_linf[sample_index]),
                "gain_l2": float(delta_l2[sample_index] / input_l2[sample_index]),
                "gain_rms": float(delta_rms[sample_index] / input_rms[sample_index]),
                "gain_linf": float(delta_linf[sample_index] / input_linf[sample_index]),
                "relative_l2": float(delta_l2[sample_index] / clean_l2[sample_index].clamp_min(1e-12)),
                "activation_cosine": float(cosine[sample_index]),
                "cosine_displacement": float(1.0 - cosine[sample_index]),
                "margin_gradient_alignment": float(alignment[sample_index]),
                "predicted_margin_drop": float(predicted_drop[sample_index]),
                "harm_per_delta_l2": float(harm_per_l2[sample_index]),
                "clean_margin": float(clean_margin[sample_index]),
                "adv_margin": float(adv_margin[sample_index]) if layer == "final logits" else np.nan,
                "actual_margin_drop": float(clean_margin[sample_index] - adv_margin[sample_index]) if layer == "final logits" else np.nan,
                "clean_confidence": float(clean_conf[sample_index]),
                "adv_confidence": float(adv_conf[sample_index]) if layer == "final logits" else np.nan,
                "confidence_drop": float(clean_conf[sample_index] - adv_conf[sample_index]) if layer == "final logits" else np.nan,
                "boundary_crossed": bool(adv_margin[sample_index] < 0) if layer == "final logits" else False,
                "state_fidelity": float(fidelity[sample_index]),
                "state_trace_distance": float(trace_distance[sample_index]),
            })
    return rows


def _matched_random_images(clean, reference_adv, epsilon: float, seed: int):
    """Generate random directions matched to each reference perturbation's L2 norm.

    Pixel bounds and the requested L-infinity radius are enforced.  A bounded
    bisection matches the post-clipping L2 norm whenever that norm is reachable.
    """

    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    direction = direction / direction.reshape(len(clean), -1).abs().amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
    target = (reference_adv - clean).reshape(len(clean), -1).norm(dim=1)
    low = torch.zeros(len(clean), dtype=clean.dtype)
    high = torch.full((len(clean),), max(epsilon * 64.0, 1e-3), dtype=clean.dtype)
    for _ in range(32):
        mid = (low + high) / 2.0
        delta = (mid.view(-1, 1, 1, 1) * direction).clamp(-epsilon, epsilon)
        candidate = (clean + delta).clamp(0.0, 1.0)
        norm = (candidate - clean).reshape(len(clean), -1).norm(dim=1)
        low = torch.where(norm < target, mid, low)
        high = torch.where(norm >= target, mid, high)
    delta = (high.view(-1, 1, 1, 1) * direction).clamp(-epsilon, epsilon)
    return (clean + delta).clamp(0.0, 1.0)


def load_audit_images(
    attack_dir: str | Path,
    seed: int,
    positions: np.ndarray,
    source_model: str,
    epsilon: float,
):
    """Load saved uint8 audit adversarial images for one original source."""

    import torch

    archive = np.load(
        Path(attack_dir) / f"seed_{seed}" / "attacker_audit_adv" / f"source={safe_model_name(source_model)}.npz"
    )
    epsilons = archive["epsilons"].astype(float)
    matches = np.flatnonzero(np.isclose(epsilons, epsilon))
    if len(matches) != 1:
        raise KeyError(f"epsilon={epsilon} not found for source {source_model}")
    images = archive["adv_images_uint8"][int(matches[0]), positions]
    return torch.from_numpy(images).to(torch.float32).unsqueeze(1) / 255.0


def validate_checkpoint_alignment(
    model,
    spec: ModelTraceSpec,
    clean_images,
    attack_dir: str | Path,
    seed: int,
    dataset_indices: np.ndarray,
) -> dict[str, object]:
    """Compare loaded checkpoint outputs with saved float16 attack baselines."""

    import torch

    with torch.no_grad():
        output = model(clean_images).detach().to(torch.float32).cpu().numpy()
    saved = np.load(
        Path(attack_dir) / f"seed_{seed}" / "baselines" / safe_model_name(spec.attack_name) / "logits_clean.npy"
    )[dataset_indices].astype(np.float32)
    return {
        "model_key": spec.key,
        "model": spec.display_name,
        "attack_target": spec.attack_name,
        "max_abs_logit_error": float(np.max(np.abs(output - saved))),
        "mean_abs_logit_error": float(np.mean(np.abs(output - saved))),
        "prediction_match": float(np.mean(output.argmax(axis=1) == saved.argmax(axis=1))),
        "n_samples": int(len(output)),
        "alignment": "exact" if np.max(np.abs(output - saved)) < 0.02 and np.mean(output.argmax(axis=1) == saved.argmax(axis=1)) == 1.0 else "near-match",
        "artifact_note": spec.artifact_note,
    }


def trace_models(
    project_root: str | Path,
    attack_dir: str | Path,
    specs: Sequence[ModelTraceSpec],
    *,
    seed: int = 0,
    epsilons: Sequence[float] = (0.05, 0.10, 0.30),
    representative_sources: Mapping[str, str] = REPRESENTATIVE_SOURCES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trace matched self, transfer, and random perturbations through models."""

    import torch

    positions, dataset_indices, label_values = common_correct_audit_positions(attack_dir, specs, seed=seed)
    if not len(positions):
        raise ValueError("No audit images are clean-correct for every selected target")
    seed_dir = Path(attack_dir) / f"seed_{seed}"
    clean_uint8 = np.load(seed_dir / "audit_clean_images_uint8.npy")[positions]
    clean_images = torch.from_numpy(clean_uint8).to(torch.float32).unsqueeze(1) / 255.0
    labels = torch.from_numpy(label_values).to(torch.long)

    records: list[dict[str, object]] = []
    alignments: list[dict[str, object]] = []
    maps: list[dict[str, object]] = []
    direction_records: list[dict[str, object]] = []
    rank_records: list[dict[str, object]] = []
    for spec in tqdm(specs, desc="Target models"):
        model, checkpoint = load_traced_model(spec, project_root, seed=seed)
        align = validate_checkpoint_alignment(model, spec, clean_images, attack_dir, seed, dataset_indices)
        align["checkpoint"] = str(checkpoint)
        alignments.append(align)
        clean_reference = capture_clean_reference(model, spec, clean_images, labels)
        activation_cache: dict[tuple[str, float], Mapping[str, object]] = {}

        for layer, activation in clean_reference["activations"].items():
            if layer == "raw input":
                meta = BlockCheckpoint("", layer, "identity", "raw pixels", 0.0)
            elif layer == "final logits":
                meta = BlockCheckpoint("", layer, "trained decision", "decision", 99.0)
            else:
                meta = next(block for block in spec.blocks if block.label == layer)
            maps.append({
                "model_key": spec.key, "model": spec.display_name, "family": spec.family,
                "architecture": spec.architecture, "layer": layer, "layer_order": meta.order,
                "path": meta.path, "shape": "x".join(map(str, activation.shape[1:])),
                "width": int(np.prod(activation.shape[1:])), "transform": meta.transform,
                "scope": meta.scope, "note": meta.note,
            })
        if spec.amplitude_source:
            source_activation = clean_reference["activations"][spec.amplitude_source]
            state = amplitude_embedding_state(source_activation, spec.amplitude_blocks)
            source_meta = next(
                (block for block in spec.blocks if block.label == spec.amplitude_source),
                BlockCheckpoint("", "", "", "", 0.0),
            )
            maps.append({
                "model_key": spec.key, "model": spec.display_name, "family": spec.family,
                "architecture": spec.architecture, "layer": "amplitude state",
                "layer_order": source_meta.order + 0.5, "path": "main",
                "shape": str(state.shape[1]), "width": int(state.shape[1]),
                "transform": "normalize + zero-pad amplitude embedding",
                "scope": "global quantum state",
                "note": "State fidelity is unchanged by the subsequent noiseless unitary.",
            })

        sources = [("self", spec.attack_name), *representative_sources.items()]
        seen: set[str] = set()
        for condition, source_model in sources:
            if source_model in seen:
                continue
            seen.add(source_model)
            for epsilon in epsilons:
                adv_images = load_audit_images(attack_dir, seed, positions, source_model, float(epsilon))
                adv_activations = capture_activations(model, spec, adv_images)
                condition_label = "self" if source_model == spec.attack_name else condition
                activation_cache[(condition_label, float(epsilon))] = adv_activations
                records.extend(_metric_rows(
                    spec, clean_reference, adv_activations, labels,
                    condition_label,
                    source_model, float(epsilon),
                ))

        for epsilon in epsilons:
            self_adv = load_audit_images(attack_dir, seed, positions, spec.attack_name, float(epsilon))
            random_images = _matched_random_images(
                clean_images, self_adv, float(epsilon), seed=seed * 1000 + int(round(epsilon * 1000)) + sum(map(ord, spec.key))
            )
            random_activations = capture_activations(model, spec, random_images)
            activation_cache[("random", float(epsilon))] = random_activations
            records.extend(_metric_rows(
                spec, clean_reference, random_activations, labels,
                "random", "random L2-matched", float(epsilon),
            ))
        directions, ranks = _direction_metrics(spec, clean_reference, activation_cache)
        direction_records.extend(directions)
        rank_records.extend(ranks)
        del model

    traces = pd.DataFrame.from_records(records)
    alignment_table = pd.DataFrame.from_records(alignments)
    architecture_map = pd.DataFrame.from_records(maps).drop_duplicates()
    direction_table = pd.DataFrame.from_records(direction_records)
    rank_table = pd.DataFrame.from_records(rank_records)
    return traces, alignment_table, architecture_map, direction_table, rank_table


def _direction_metrics(
    spec: ModelTraceSpec,
    clean_reference: Mapping[str, object],
    activation_cache: Mapping[tuple[str, float], Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Pairwise source-direction similarity and displacement effective rank."""

    import torch

    clean_acts = dict(clean_reference["activations"])
    conditions_by_epsilon: dict[float, list[str]] = {}
    for condition, epsilon in activation_cache:
        conditions_by_epsilon.setdefault(float(epsilon), []).append(condition)
    pair_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    for epsilon, conditions in conditions_by_epsilon.items():
        conditions = sorted(set(conditions))
        layers = list(clean_acts)
        for layer in layers:
            clean = clean_acts[layer].reshape(clean_acts[layer].shape[0], -1).to(torch.float32)
            deltas: dict[str, object] = {}
            for condition in conditions:
                adv = activation_cache[(condition, epsilon)][layer]
                deltas[condition] = adv.reshape(adv.shape[0], -1).to(torch.float32) - clean
            for index, source_a in enumerate(conditions):
                for source_b in conditions[index + 1 :]:
                    delta_a = deltas[source_a]
                    delta_b = deltas[source_b]
                    cosine = torch.nn.functional.cosine_similarity(delta_a, delta_b, dim=1, eps=1e-12)
                    sign = (delta_a.sign() == delta_b.sign()).to(torch.float32).mean(dim=1)
                    pair_rows.append({
                        "model_key": spec.key, "model": spec.display_name,
                        "family": spec.family, "architecture": spec.architecture,
                        "epsilon": float(epsilon), "layer": layer,
                        "source_a": source_a, "source_b": source_b,
                        "cosine_mean": float(cosine.mean()), "cosine_median": float(cosine.median()),
                        "cosine_q25": float(torch.quantile(cosine, 0.25)),
                        "cosine_q75": float(torch.quantile(cosine, 0.75)),
                        "sign_agreement_mean": float(sign.mean()), "n_samples": int(len(cosine)),
                    })

            matrix = torch.cat([deltas[condition] for condition in conditions], dim=0)
            matrix = matrix - matrix.mean(dim=0, keepdim=True)
            gram = matrix @ matrix.T
            eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
            energy = eigenvalues.sum()
            if float(energy) > 0:
                effective_rank = float(energy.square() / eigenvalues.square().sum().clamp_min(1e-12))
                descending = torch.flip(eigenvalues, dims=(0,))
                rank90 = int(torch.searchsorted(descending.cumsum(0), 0.9 * energy).item() + 1)
            else:
                effective_rank = 0.0
                rank90 = 0
            rank_rows.append({
                "model_key": spec.key, "model": spec.display_name,
                "family": spec.family, "architecture": spec.architecture,
                "epsilon": float(epsilon), "layer": layer,
                "effective_rank": effective_rank, "rank90": rank90,
                "matrix_rows": int(matrix.shape[0]), "activation_width": int(matrix.shape[1]),
                "n_conditions": int(len(conditions)),
            })
    return pair_rows, rank_rows


def summarize_traces(traces: pd.DataFrame) -> pd.DataFrame:
    """Median/IQR summaries used by figures and report tables."""

    traces = traces.copy()
    traces["boundary_crossed"] = traces["boundary_crossed"].astype(float)
    metrics = [
        "delta_l2", "delta_rms", "delta_linf", "gain_l2", "gain_rms", "gain_linf",
        "relative_l2", "cosine_displacement", "margin_gradient_alignment",
        "predicted_margin_drop", "harm_per_delta_l2", "actual_margin_drop",
        "clean_margin", "adv_margin", "clean_confidence", "adv_confidence",
        "confidence_drop", "state_fidelity", "state_trace_distance", "boundary_crossed",
    ]
    grouped = traces.groupby(
        ["model_key", "model", "family", "architecture", "condition", "source_model", "epsilon",
         "layer", "layer_order", "path", "transform", "scope", "shape", "width"],
        dropna=False,
    )
    pieces = []
    for metric in metrics:
        summary = grouped[metric].agg(
            **{f"{metric}_median": "median", f"{metric}_q25": lambda s: s.quantile(0.25), f"{metric}_q75": lambda s: s.quantile(0.75), f"{metric}_mean": "mean"}
        )
        pieces.append(summary)
    return pd.concat(pieces, axis=1).reset_index()


def find_norm_harm_counterexamples(summary: pd.DataFrame, epsilon: float = 0.10) -> pd.DataFrame:
    """Find consecutive main-path stages where norm and decision visibility disagree."""

    work = summary[np.isclose(summary["epsilon"], epsilon)].copy()
    work = work[(work["path"].isin(["main", "merge"])) & work["predicted_margin_drop_median"].notna()]
    rows: list[dict[str, object]] = []
    for keys, frame in work.groupby(["model_key", "model", "condition"], sort=False):
        frame = frame.sort_values("layer_order")
        prior = None
        for current in frame.to_dict("records"):
            if prior is not None:
                norm_ratio = current["delta_rms_median"] / max(prior["delta_rms_median"], 1e-12)
                prior_harm = abs(prior["harm_per_delta_l2_median"])
                current_harm = abs(current["harm_per_delta_l2_median"])
                harm_ratio = current_harm / max(prior_harm, 1e-12)
                if (norm_ratio < 0.8 and harm_ratio > 1.25) or (norm_ratio > 1.25 and harm_ratio < 0.8):
                    rows.append({
                        "model_key": keys[0], "model": keys[1], "condition": keys[2],
                        "epsilon": float(epsilon), "from_layer": prior["layer"], "to_layer": current["layer"],
                        "rms_norm_ratio": float(norm_ratio), "decision_visibility_ratio": float(harm_ratio),
                        "pattern": "norm down, visibility up" if norm_ratio < 1 else "norm up, visibility down",
                    })
            prior = current
    return pd.DataFrame(rows).sort_values(
        ["pattern", "decision_visibility_ratio"], ascending=[True, False]
    ) if rows else pd.DataFrame()


def select_attack_models(variant_seed: pd.DataFrame, specs: Sequence[ModelTraceSpec]) -> pd.DataFrame:
    names = {spec.attack_name for spec in specs}
    return variant_seed[variant_seed["model"].isin(names)].copy()


_STAGE_LAYERS: Mapping[str, Mapping[str, str | None]] = {
    "cnn": {"first": "conv 1", "encoder": "pool 2", "quantum": None, "head": "final logits"},
    "resnet18": {"first": "stem convolution", "encoder": "residual stage 4", "quantum": None, "head": "final logits"},
    "mlp": {"first": "dense 784->128", "encoder": "dense 128->64", "quantum": None, "head": "final logits"},
    "mlpmixer": {"first": "patch projection", "encoder": "mixer block 4", "quantum": None, "head": "final logits"},
    "hvql": {"first": "conv 1", "encoder": "pre-quantum latent", "quantum": "measured quantum vector", "head": "final logits"},
    "hpvql": {"first": "conv 1", "encoder": "pre-quantum latent", "quantum": "measured quantum vector", "head": "final logits"},
    "hvqm": {"first": "conv 1", "encoder": "encoder output", "quantum": "measured quantum logits", "head": "final logits"},
    "hqe": {"first": "dense 784->4096", "encoder": "encoder latent 1024", "quantum": "measured quantum vector", "head": "final logits"},
    "hquanv": {"first": "quanvolution map", "encoder": "flattened quanvolution", "quantum": "quanvolution map", "head": "final logits"},
    "drophvql": {"first": "conv 1", "encoder": "inference dropout", "quantum": "measured quantum vector", "head": "final logits"},
    "reshvql": {"first": "conv 1", "encoder": "pre-quantum latent", "quantum": "quantum branch", "head": "final logits"},
    "vqm50": {"first": "amplitude state", "encoder": None, "quantum": "measured observables", "head": "final logits"},
    "vqm100": {"first": "amplitude state", "encoder": None, "quantum": "measured observables", "head": "final logits"},
    "vqm200": {"first": "amplitude state", "encoder": None, "quantum": "measured observables", "head": "final logits"},
}


def build_stage_effect_table(trace_summary: pd.DataFrame, epsilon: float = 0.10) -> pd.DataFrame:
    """Build the requested model-by-stage qualitative table from measured ratios."""

    work = trace_summary[
        np.isclose(trace_summary["epsilon"], epsilon) & (trace_summary["condition"] == "self")
    ].copy()
    rows: list[dict[str, object]] = []

    def describe(frame: pd.DataFrame, layer: str | None, previous: str | None) -> str:
        if layer is None:
            return "not applicable"
        current = frame[frame["layer"] == layer]
        prior = frame[frame["layer"] == previous] if previous else frame[frame["layer"] == "raw input"]
        if current.empty or prior.empty:
            return "not measured"
        c = current.iloc[0]
        p = prior.iloc[0]
        norm_ratio = float(c["delta_rms_median"]) / max(float(p["delta_rms_median"]), 1e-12)
        c_visibility = abs(float(c["harm_per_delta_l2_median"]))
        p_visibility = abs(float(p["harm_per_delta_l2_median"]))
        if np.isfinite(c_visibility) and np.isfinite(p_visibility):
            visibility = f"; visibility x{c_visibility / max(p_visibility, 1e-12):.2f}"
        else:
            visibility = ""
        direction = "attenuates" if norm_ratio < 0.9 else "amplifies" if norm_ratio > 1.1 else "preserves"
        return f"{direction} RMS (x{norm_ratio:.2f}){visibility}"

    for model_key, frame in work.groupby("model_key", sort=False):
        frame = frame.sort_values("layer_order")
        stage = _STAGE_LAYERS.get(model_key, {})
        first = stage.get("first")
        encoder = stage.get("encoder")
        quantum = stage.get("quantum")
        head = stage.get("head")
        measured = frame[
            frame["predicted_margin_drop_median"].notna()
            & frame["path"].isin(["main", "merge"])
        ].sort_values("layer_order").copy()
        measured["harm"] = measured["predicted_margin_drop_median"].abs()
        measured["harm_increase"] = measured["harm"].diff()
        downstream = measured[measured["layer"] != "raw input"]
        if downstream.empty:
            dominant = "not measured"
        else:
            peak = downstream.loc[downstream["harm_increase"].idxmax()]
            if float(peak["harm_increase"]) > 0:
                dominant = f"{peak['layer']} (largest downstream margin-harm increase)"
            else:
                dominant = "input-aligned; no downstream concentration"
        rows.append({
            "Model": frame["model"].iloc[0],
            "First block effect": describe(frame, first, "raw input"),
            "Encoder effect": describe(frame, encoder, first),
            "Quantum block effect": describe(frame, quantum, encoder or first),
            "Final head effect": describe(frame, head, quantum or encoder or first),
            "Dominant failure stage": dominant,
            "epsilon": float(epsilon),
            "cohort": "common clean-correct audit subset",
        })
    return pd.DataFrame(rows)


def _save_figure(fig, output_path: str | Path) -> dict[str, Path]:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for suffix in (".png", ".pdf"):
        target = path.with_suffix(suffix)
        fig.savefig(target, dpi=220, bbox_inches="tight", facecolor="white")
        paths[suffix[1:]] = target
    return paths


def _plot_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "axes.edgecolor": "#4b5563", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "#e5e7eb", "grid.linewidth": 0.7,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def plot_attack_outcomes(
    variant_seed: pd.DataFrame,
    specs: Sequence[ModelTraceSpec],
    output_path: str | Path,
):
    """Four-panel clean, weak-regime, sweep, and transfer metric figure."""

    import matplotlib.pyplot as plt

    _plot_style()
    selected = select_attack_models(variant_seed, specs)
    labels = {spec.attack_name: spec.display_name for spec in specs}
    grouped = selected.groupby("model", as_index=False).mean(numeric_only=True)
    grouped["label"] = grouped["model"].map(labels)
    grouped = grouped.dropna(subset=["label"]).sort_values("robust_accuracy_010")
    colors = ["#335c81" if "HVQL" in label or "HPVQL" in label else "#d97706" if "VQM" in label else "#6b7280" for label in grouped["label"]]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    y = np.arange(len(grouped))
    axes[0, 0].barh(y - 0.18, grouped["robust_accuracy_000"], height=0.34, color="#cbd5e1", label="clean")
    axes[0, 0].barh(y + 0.18, grouped["robust_accuracy_010"], height=0.34, color=colors, label="FGSM epsilon=0.10")
    axes[0, 0].set_yticks(y, grouped["label"])
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_title("Clean and weak-regime robust accuracy")
    axes[0, 0].set_xlabel("Accuracy")
    axes[0, 0].legend(frameon=False, loc="lower right")

    axes[0, 1].barh(y, grouped["retention_010"], color=colors)
    axes[0, 1].set_yticks(y, grouped["label"])
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_title("Accuracy retention at epsilon=0.10")
    axes[0, 1].set_xlabel("Robust accuracy / clean accuracy")

    axes[1, 0].barh(y, grouped["robust_accuracy_auc"], color=colors)
    axes[1, 0].set_yticks(y, grouped["label"])
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_title("Full-sweep robust-accuracy AUC")
    axes[1, 0].set_xlabel("AUC over epsilon in [0,1]")

    axes[1, 1].scatter(grouped["incoming_asr_010"], grouped["outgoing_asr_010"], c=colors, s=65, edgecolor="#1f2937")
    focus_offsets = {
        "CNN": (5, -12), "HVQL (raw artifact: HVQM, Strong d=50)": (-78, -12),
        "HPVQL (5 blocks, d=50)": (-72, 6), "HQE": (6, -13),
        "HQuanv": (5, 6), "ResNet18": (5, 6), "MLP (128-64)": (5, 6),
        "MLPMixer": (5, -13), "HVQM (raw artifact: HVQL, Strong d=50)": (-82, 7),
    }
    for _, row in grouped.iterrows():
        if row["label"] not in focus_offsets:
            continue
        axes[1, 1].annotate(
            row["label"], (row["incoming_asr_010"], row["outgoing_asr_010"]),
            xytext=focus_offsets[row["label"]], textcoords="offset points", fontsize=7,
        )
    axes[1, 1].set_xlim(left=0)
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].set_title("Black-box exposure and source potency at epsilon=0.10")
    axes[1, 1].set_xlabel("Incoming mean ASR (target exposure)")
    axes[1, 1].set_ylabel("Outgoing mean ASR (source potency)")
    paths = _save_figure(fig, output_path)
    return fig, paths


def plot_block_diagrams(
    architecture_map: pd.DataFrame,
    output_path: str | Path,
):
    """Render one compact measured-shape block diagram per selected model."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    _plot_style()
    models = list(architecture_map["model"].drop_duplicates())
    fig, axes = plt.subplots(len(models), 1, figsize=(24, max(2.0 * len(models), 8)), squeeze=False)
    palette = {"raw pixels": "#e5e7eb", "decision": "#f8d7a6", "global quantum state": "#d8c7ee"}
    for axis, model_name in zip(axes[:, 0], models):
        frame = architecture_map[architecture_map["model"] == model_name].sort_values("layer_order")
        axis.set_xlim(0, max(len(frame), 1))
        axis.set_ylim(0, 1)
        axis.axis("off")
        axis.text(-0.02, 0.5, model_name, ha="right", va="center", fontsize=9, fontweight="bold", transform=axis.transAxes)
        for position, (_, row) in enumerate(frame.iterrows()):
            scope = str(row["scope"])
            fill = palette.get(scope, "#c7dcef" if "local" in scope or "hierarchical" in scope else "#f3d6a3" if "global" in scope else "#e5e7eb")
            box = FancyBboxPatch((position + 0.05, 0.2), 0.80, 0.60, boxstyle="round,pad=0.02", facecolor=fill, edgecolor="#334155", linewidth=0.8)
            axis.add_patch(box)
            label = f"{row['layer']}\n{row['shape']}"
            axis.text(position + 0.45, 0.5, label, ha="center", va="center", fontsize=6.5)
            if position < len(frame) - 1:
                axis.annotate("", xy=(position + 1.03, 0.5), xytext=(position + 0.87, 0.5), arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 0.8})
    paths = _save_figure(fig, output_path)
    return fig, paths


def plot_layerwise_gain(
    trace_summary: pd.DataFrame,
    model_keys: Sequence[str],
    output_path: str | Path,
    *,
    epsilon: float = 0.10,
    conditions: Sequence[str] = ("self", "CNN", "MLP", "HQuanv", "HVQL", "VQM50", "random"),
):
    """Plot dimension-normalized RMS perturbation gain by block."""

    import matplotlib.pyplot as plt

    _plot_style()
    frame = trace_summary[
        trace_summary["model_key"].isin(model_keys)
        & np.isclose(trace_summary["epsilon"], epsilon)
        & trace_summary["condition"].isin(conditions)
    ].copy()
    ncols = 2
    nrows = int(math.ceil(len(model_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.8 * nrows), squeeze=False)
    palette = {"self": "#1f4e79", "CNN": "#d97706", "MLP": "#9a5b13", "HQuanv": "#7a8f2a", "HVQL": "#7c3f96", "VQM50": "#d46a92", "random": "#6b7280"}
    for axis, model_key in zip(axes.flat, model_keys):
        model_frame = frame[(frame["model_key"] == model_key) & (frame["path"].isin(["main", "merge"]))]
        ordered_layers = (
            model_frame[["layer", "layer_order"]]
            .drop_duplicates()
            .sort_values("layer_order")["layer"]
            .tolist()
        )
        positions = {layer: position for position, layer in enumerate(ordered_layers)}
        for condition in conditions:
            line = model_frame[model_frame["condition"] == condition].sort_values("layer_order")
            if line.empty:
                continue
            x = line["layer"].map(positions)
            axis.plot(x, line["gain_rms_median"], marker="o", ms=3.5, lw=1.4, label=condition, color=palette.get(condition, "#64748b"))
        axis.set_xticks(range(len(ordered_layers)), ordered_layers, rotation=52, ha="right", fontsize=7)
        axis.set_yscale("log")
        axis.axhline(1.0, color="#111827", lw=0.8, ls="--")
        title = model_frame["model"].iloc[0] if not model_frame.empty else model_key
        axis.set_title(title)
        axis.set_ylabel("RMS displacement gain vs input")
        axis.set_xlabel("Measured block")
    for axis in axes.flat[len(model_keys):]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=min(len(labels), 7), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    paths = _save_figure(fig, output_path)
    return fig, paths


def plot_margin_trajectories(
    trace_summary: pd.DataFrame,
    model_keys: Sequence[str],
    output_path: str | Path,
    *,
    epsilon: float = 0.10,
    conditions: Sequence[str] = ("self", "CNN", "HVQL", "VQM50", "random"),
):
    """Plot downstream-linearized margins, with the final point exact."""

    import matplotlib.pyplot as plt

    _plot_style()
    frame = trace_summary[
        trace_summary["model_key"].isin(model_keys)
        & np.isclose(trace_summary["epsilon"], epsilon)
        & trace_summary["condition"].isin(conditions)
        & trace_summary["path"].isin(["main", "merge"])
    ].copy()
    frame["margin_trajectory"] = frame["clean_margin_median"] - frame["predicted_margin_drop_median"]
    final = frame["layer"] == "final logits"
    frame.loc[final, "margin_trajectory"] = frame.loc[final, "adv_margin_median"]
    ncols = 2
    nrows = int(math.ceil(len(model_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.8 * nrows), squeeze=False)
    palette = {"self": "#1f4e79", "CNN": "#d97706", "HVQL": "#7c3f96", "VQM50": "#d46a92", "random": "#6b7280"}
    for axis, model_key in zip(axes.flat, model_keys):
        model_frame = frame[frame["model_key"] == model_key]
        ordered_layers = (
            model_frame[["layer", "layer_order"]]
            .drop_duplicates()
            .sort_values("layer_order")["layer"]
            .tolist()
        )
        positions = {layer: position for position, layer in enumerate(ordered_layers)}
        for condition in conditions:
            line = model_frame[model_frame["condition"] == condition].dropna(subset=["margin_trajectory"]).sort_values("layer_order")
            if line.empty:
                continue
            x = line["layer"].map(positions)
            axis.plot(x, line["margin_trajectory"], marker="o", ms=3.5, lw=1.4, label=condition, color=palette.get(condition, "#64748b"))
        axis.set_xticks(range(len(ordered_layers)), ordered_layers, rotation=52, ha="right", fontsize=7)
        axis.axhline(0.0, color="#111827", lw=0.8, ls="--")
        title = model_frame["model"].iloc[0] if not model_frame.empty else model_key
        axis.set_title(title)
        axis.set_ylabel("Margin (linearized internally; exact at logits)")
        axis.set_xlabel("Measured block")
    for axis in axes.flat[len(model_keys):]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=min(len(labels), 5), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    paths = _save_figure(fig, output_path)
    return fig, paths


def plot_source_target_asr(
    attack_table: pd.DataFrame,
    specs: Sequence[ModelTraceSpec],
    output_path: str | Path,
    *,
    epsilon: float = 0.10,
):
    """Heatmap of exact full-dataset ASR for selected source-target pairs."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    _plot_style()
    names = [spec.attack_name for spec in specs]
    labels = {spec.attack_name: spec.display_name for spec in specs}
    frame = attack_table[
        attack_table["source_model"].isin(names)
        & attack_table["target_model"].isin(names)
        & np.isclose(attack_table["epsilon"], epsilon)
    ]
    matrix = frame.groupby(["source_model", "target_model"])["ASR"].mean().unstack()
    matrix = matrix.reindex(index=names, columns=names).rename(index=labels, columns=labels)
    fig, axis = plt.subplots(figsize=(16, 13), constrained_layout=True)
    sns.heatmap(matrix, ax=axis, cmap=sns.light_palette("#1f4e79", as_cmap=True), vmin=0, vmax=max(0.5, float(np.nanmax(matrix.to_numpy()))), annot=True, fmt=".2f", square=True, cbar_kws={"label": "ASR"}, annot_kws={"fontsize": 6})
    axis.set_title(f"Selected source-target attack success rates at epsilon={epsilon:.2f}")
    axis.set_xlabel("Target model")
    axis.set_ylabel("Source model")
    paths = _save_figure(fig, output_path)
    return fig, paths, matrix


def write_analysis_tables(
    output_dir: str | Path,
    *,
    attack_table: pd.DataFrame,
    variant_seed: pd.DataFrame,
    architecture_summary: pd.DataFrame,
    traces: pd.DataFrame,
    trace_summary: pd.DataFrame,
    alignments: pd.DataFrame,
    architecture_map: pd.DataFrame,
    direction_similarity: pd.DataFrame,
    effective_rank: pd.DataFrame,
    stage_effects: pd.DataFrame,
    counterexamples: pd.DataFrame,
) -> dict[str, Path]:
    """Write reproducible CSV tables used by the notebook and report."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "attack_aggregates": attack_table,
        "variant_seed_metrics": variant_seed,
        "architecture_metrics": architecture_summary,
        "layerwise_trace_samples": traces,
        "layerwise_trace_summary": trace_summary,
        "checkpoint_alignment": alignments,
        "architecture_map": architecture_map,
        "source_direction_similarity": direction_similarity,
        "perturbation_effective_rank": effective_rank,
        "stage_effects": stage_effects,
        "norm_harm_counterexamples": counterexamples,
    }
    paths = {}
    for name, frame in tables.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


__all__ = [
    "DEFAULT_MODEL_ROOT", "DEFAULT_ATTACK_DIR", "DEFAULT_OUTPUT_DIR",
    "BlockCheckpoint", "ModelTraceSpec", "REPRESENTATIVE_SOURCES",
    "default_model_specs", "safe_model_name", "crawl_model_dirs",
    "load_attack_aggregates", "canonical_architecture", "model_family",
    "summarize_attack_metrics", "common_correct_audit_positions",
    "load_traced_model", "capture_clean_reference", "capture_activations",
    "amplitude_embedding_state", "load_audit_images", "validate_checkpoint_alignment",
    "trace_models", "summarize_traces",
    "find_norm_harm_counterexamples", "select_attack_models", "write_analysis_tables",
    "build_stage_effect_table", "plot_attack_outcomes", "plot_block_diagrams",
    "plot_layerwise_gain", "plot_margin_trajectories", "plot_source_target_asr",
]

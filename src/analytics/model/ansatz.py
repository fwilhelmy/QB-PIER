from __future__ import annotations

"""Render circuit diagrams for registered ansatz definitions at a chosen depth."""

from types import SimpleNamespace
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import yaml

from analytics.paths import find_project_root
from qdevice import CIRCUIT_STYLE
from layers.quantum.ansatz import AnsatzFactory
from layers.quantum.ansatz.ResidualAnsatzWrapper import ResidualAnsatzWrapper

log = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _default_args() -> SimpleNamespace:
    return SimpleNamespace(
        depth=2,
        output_dir="outputs/ansatz_circuits",
        n_qubits=4,
        embedding="none",
        expanded=False,
        dpi=220,
        formats=["png"],
        ansatz=None,
        residual_base="strong",
        verbose=True,
    )


def _ansatz_conf_dir() -> Path:
    return find_project_root() / "conf" / "model" / "ansatz"


def _resolve_input_size(n_qubits: int, embedding: str) -> int:
    if n_qubits <= 0:
        raise ValueError("--n-qubits must be positive.")
    if embedding == "amplitude":
        return 2 ** n_qubits
    return n_qubits


def _load_ansatz_definitions(conf_dir: Path) -> List[Dict[str, Any]]:
    definitions: List[Dict[str, Any]] = []

    for path in sorted(conf_dir.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        definitions.append(
            {
                "alias": path.stem,
                "class": cfg.get("class"),
                "params": cfg.get("params") or {},
                "path": str(path),
            }
        )

    return definitions


def _build_default_inputs(ansatz: Any, embedding_name: str) -> np.ndarray:
    if embedding_name == "none" or getattr(ansatz, "embedding", None) is None:
        return np.zeros(ansatz.n_qubits, dtype=float)

    embedding = getattr(ansatz, "embedding", None)
    input_size = int(getattr(embedding, "input_size", ansatz.n_qubits))
    embedding_type = type(embedding).__name__.lower()

    if "amplitude" in embedding_type:
        x = np.zeros(input_size, dtype=float)
        x[0] = 1.0
        return x

    return np.zeros(input_size, dtype=float)


def _build_default_kwargs(ansatz: Any) -> Dict[str, np.ndarray]:
    kwargs: Dict[str, np.ndarray] = {}
    for name, shape in (getattr(ansatz, "weight_shapes", None) or {}).items():
        kwargs[name] = np.zeros(shape, dtype=float)
    return kwargs


def _extract_resources(specs: Any) -> tuple[int | None, int | None]:
    resources = getattr(specs, "resources", None)
    if resources is None and isinstance(specs, dict):
        resources = specs.get("resources")

    num_gates = None
    depth = None

    if resources is not None:
        num_gates = getattr(resources, "num_gates", None)
        depth = getattr(resources, "depth", None)

    if num_gates is None and isinstance(specs, dict):
        num_gates = specs.get("num_operations")
    if depth is None and isinstance(specs, dict):
        depth = specs.get("depth")

    return (
        int(num_gates) if num_gates is not None else None,
        int(depth) if depth is not None else None,
    )


def _resolve_concrete_base(
    alias_to_definition: Dict[str, Dict[str, Any]],
    residual_base_alias: str,
) -> tuple[type[Any], Dict[str, Any], str]:
    if residual_base_alias not in alias_to_definition:
        available = ", ".join(sorted(alias_to_definition))
        raise KeyError(
            f"Unknown residual base ansatz '{residual_base_alias}'. Available aliases: {available}"
        )

    base_def = alias_to_definition[residual_base_alias]
    base_class_name = str(base_def["class"])
    if base_class_name == "ResidualAnsatzWrapper":
        raise ValueError("The residual base ansatz cannot itself be the residual wrapper.")

    base_cls = AnsatzFactory._get(base_class_name)
    return base_cls, dict(base_def.get("params") or {}), base_class_name


def _instantiate_ansatz(
    definition: Dict[str, Any],
    *,
    n_layers: int,
    n_qubits: int,
    embedding: str,
    residual_base_alias: str,
    alias_to_definition: Dict[str, Dict[str, Any]],
) -> tuple[Any, Dict[str, Any]]:
    alias = str(definition["alias"])
    class_name = str(definition["class"])
    params = dict(definition.get("params") or {})
    input_size = _resolve_input_size(n_qubits=n_qubits, embedding=embedding)

    meta: Dict[str, Any] = {
        "alias": alias,
        "class": class_name,
        "input_size": input_size,
        "n_qubits_requested": n_qubits,
        "embedding_requested": embedding,
        "residual_base_alias": None,
        "residual_base_class": None,
    }

    if class_name == "ResidualAnsatzWrapper":
        base_cls, base_params, base_class_name = _resolve_concrete_base(
            alias_to_definition=alias_to_definition,
            residual_base_alias=residual_base_alias,
        )
        wrapped_cls = ResidualAnsatzWrapper(base_cls)
        ansatz = wrapped_cls(
            input_size=input_size,
            n_layers=n_layers,
            **base_params,
        )
        meta["class"] = wrapped_cls.__name__
        meta["residual_base_alias"] = residual_base_alias
        meta["residual_base_class"] = base_class_name
        meta["embedding_effective"] = "none"
        return ansatz, meta

    ansatz = AnsatzFactory.create(
        class_name,
        input_size=input_size,
        n_layers=n_layers,
        embedding=embedding,
        **params,
    )
    meta["embedding_effective"] = embedding
    return ansatz, meta


def _render_figure(
    ansatz: Any,
    *,
    embedding_name: str,
    expanded: bool,
    dpi: int,
    output_base: Path,
    formats: Iterable[str],
    title: str,
) -> tuple[List[str], int | None, int | None]:
    default_inputs = _build_default_inputs(ansatz, embedding_name)
    default_kwargs = _build_default_kwargs(ansatz)

    draw_level = "device" if expanded else "gradient"
    drawer = qml.draw_mpl(ansatz.qnode, style=CIRCUIT_STYLE, level=draw_level)
    fig, _ = drawer(default_inputs, **default_kwargs)
    fig.suptitle(title, y=1.02)

    written_paths: List[str] = []
    for fmt in formats:
        out_path = output_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        written_paths.append(str(out_path))

    plt.close(fig)

    specs = qml.specs(ansatz.qnode, compute_depth=True, level="device")(
        default_inputs,
        **default_kwargs,
    )
    num_gates, circuit_depth = _extract_resources(specs)
    return written_paths, num_gates, circuit_depth


def main() -> None:
    args = _default_args()
    _configure_logging(args.verbose)

    if args.depth <= 0:
        raise ValueError("--depth must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    definitions = _load_ansatz_definitions(_ansatz_conf_dir())
    alias_to_definition = {str(item["alias"]): item for item in definitions}

    selected_aliases = set(args.ansatz) if args.ansatz else None
    rows: List[Dict[str, Any]] = []

    for definition in definitions:
        alias = str(definition["alias"])
        if selected_aliases is not None and alias not in selected_aliases:
            continue

        row: Dict[str, Any] = {
            "alias": alias,
            "config_class": definition.get("class"),
            "config_params": str(definition.get("params") or {}),
            "depth_requested": int(args.depth),
            "n_qubits_requested": int(args.n_qubits),
            "embedding_requested": args.embedding,
            "expanded": bool(args.expanded),
            "status": "ok",
            "error": None,
            "figure_paths": None,
            "effective_class": None,
            "effective_embedding": None,
            "effective_n_qubits": None,
            "input_size": None,
            "num_gates": None,
            "circuit_depth": None,
            "residual_base_alias": None,
            "residual_base_class": None,
        }

        try:
            ansatz, meta = _instantiate_ansatz(
                definition,
                n_layers=args.depth,
                n_qubits=args.n_qubits,
                embedding=args.embedding,
                residual_base_alias=args.residual_base,
                alias_to_definition=alias_to_definition,
            )

            title = f"{alias} | depth={args.depth} | qubits={ansatz.n_qubits} | embedding={meta['embedding_effective']}"
            output_base = output_dir / f"ansatz={alias}_depth={args.depth}_qubits={ansatz.n_qubits}"
            figure_paths, num_gates, circuit_depth = _render_figure(
                ansatz,
                embedding_name=str(meta["embedding_effective"]),
                expanded=args.expanded,
                dpi=args.dpi,
                output_base=output_base,
                formats=args.formats,
                title=title,
            )

            row.update(
                {
                    "figure_paths": " | ".join(figure_paths),
                    "effective_class": meta["class"],
                    "effective_embedding": meta["embedding_effective"],
                    "effective_n_qubits": int(ansatz.n_qubits),
                    "input_size": int(meta["input_size"]),
                    "num_gates": num_gates,
                    "circuit_depth": circuit_depth,
                    "residual_base_alias": meta["residual_base_alias"],
                    "residual_base_class": meta["residual_base_class"],
                }
            )
            log.info("Rendered %s -> %s", alias, row["figure_paths"])
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
            log.exception("Failed to render ansatz %s", alias)

        rows.append(row)

    summary_path = output_dir / "ansatz_circuit_summary.csv"
    pd.DataFrame(rows).sort_values(["status", "alias"]).to_csv(summary_path, index=False)
    log.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()

"""
Migrate attack_experiment artifacts to canonical model naming.

Copies INPUT_DIR to OUTPUT_DIR (legacy untouched), renames model references
in tables and paths, and drops bugged/incomplete models.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- configuration (toggle here; no argparse per AGENTS.md) ---
INPUT_DIR = "logs/ATTACKS/attack_experiment"
OUTPUT_DIR = "logs/ATTACKS/attack_experiment_canonical"
DRY_RUN = True

OLD_TO_NEW: dict[str, str] = {
    # Classical
    "Classical/convnet/conv_channels=[32,64],fc_sizes=[128]": (
        "Classical/CNN/conv_channels=[32,64], fc_sizes=[128]"
    ),
    "Classical/convnet/conv_channels=[8],fc_sizes=[10]": (
        "Classical/CNN/conv_channels=[8], fc_sizes=[10]"
    ),
    "Classical/convnet/conv_channels=[8],fc_sizes=[32]": (
        "Classical/CNN/conv_channels=[8], fc_sizes=[32]"
    ),
    "Classical/lenet": "Classical/CNN/LeNet",
    "Classical/resnet18": "Classical/ResNet18",
    "Classical/mlpmixer": "Classical/MLPMixer",
    "Classical/mlp/hidden_sizes=[1024, 512, 64, 64]": (
        "Classical/MLP/hidden_sizes=[1024,512,64,64]"
    ),
    "Classical/mlp/hidden_sizes=[128,64]": "Classical/MLP/hidden_sizes=[128,64]",
    "Classical/mlp/hidden_sizes=[64,32]": "Classical/MLP/hidden_sizes=[64,32]",
    "Classical/mlp/hidden_sizes=[32,16]": "Classical/MLP/hidden_sizes=[32,16]",
    "Classical/mlp/hidden_sizes=[8,4]": "Classical/MLP/hidden_sizes=[8,4]",
    "Classical/mlp/hidden_sizes=[4,4]": "Classical/MLP/hidden_sizes=[4,4]",
    "Classical/mlp/hidden_sizes=[4,2]": "Classical/MLP/hidden_sizes=[4,2]",
    # Hybrid (non HVQ*)
    "Hybrid/hqe": "Hybrid/HQE",
    "Hybrid/hquanv": "Hybrid/HQuanv",
    "Hybrid/hpvqm/qblocks=5,qdepth=50": (
        "Hybrid/HPVQL/qblocks=5, qdepth=50, ansatz=strong"
    ),
    "Hybrid/hpvqm/qblocks=5,qdepth=100": (
        "Hybrid/HPVQL/qblocks=5, qdepth=100, ansatz=strong"
    ),
    "Hybrid/hpvqm/qblocks=10,qdepth=50": (
        "Hybrid/HPVQL/qblocks=10, qdepth=50, ansatz=strong"
    ),
    "Hybrid/hpvqm/qblocks=10,qdepth=100": (
        "Hybrid/HPVQL/qblocks=10, qdepth=100, ansatz=strong"
    ),
    # HVQM <-> HVQL correction (old labels were swapped)
    "Hybrid/hvql/ansatz=random,qdepth=50": "Hybrid/HVQM/ansatz=random, qdepth=50",
    "Hybrid/hvql/ansatz=strong,qdepth=50": "Hybrid/HVQM/ansatz=strong, qdepth=50",
    "Hybrid/hvql/dropout": "Hybrid/HVQL/dropout",
    "Hybrid/hvql/residuals": "Hybrid/HVQL/residuals",
    "Hybrid/hvqm/ansatz=basic,qdepth=50": "Hybrid/HVQL/ansatz=basic, qdepth=50",
    "Hybrid/hvqm/ansatz=random,qdepth=50": "Hybrid/HVQL/ansatz=random, qdepth=50",
    "Hybrid/hvqm/ansatz=strong,qdepth=50": "Hybrid/HVQL/ansatz=strong, qdepth=50",
    "Hybrid/hvqm/ansatz=strong,qdepth=100": "Hybrid/HVQL/ansatz=strong, qdepth=100",
    "Hybrid/hvqm/ansatz=strong,qdepth=200": "Hybrid/HVQL/ansatz=strong, qdepth=200",
    # Quantum
    "Quantum/VQM50-Basic": "Quantum/VQM/ansatz=basic, qdepth=50",
    "Quantum/VQM50-NoEntangling": "Quantum/VQM/ansatz=no_entangling, qdepth=50",
    "Quantum/VQM50-Random": "Quantum/VQM/ansatz=random, qdepth=50",
    "Quantum/VQM50-Strong": "Quantum/VQM/ansatz=strong, qdepth=50",
    "Quantum/VQM100": "Quantum/VQM/ansatz=strong, qdepth=100",
    "Quantum/VQM200": "Quantum/VQM/ansatz=strong, qdepth=200",
}

DELETE_MODELS: set[str] = {
    "Hybrid/hvql/ansatz=basic,qdepth=50",
    "Quantum/VQM50-Identity",
    "Hybrid/hpvql",
}

CANONICAL_NAMES: set[str] = set(OLD_TO_NEW.values())

MODEL_COLUMNS = ("source_model", "target_model")


def _safe_name(name: str) -> str:
    return (
        name.replace("\\", "__")
        .replace("/", "__")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _project_root() -> Path:
    current = Path(__file__).resolve().parent.parent
    return current


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return _project_root() / p


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        for engine in ("pyarrow", "fastparquet", None):
            try:
                return pd.read_parquet(path, engine=engine)
            except ImportError:
                continue
        raise ImportError(
            "Reading parquet requires pyarrow or fastparquet. "
            "Install with: pip install pyarrow"
        )
    return pd.read_csv(path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        for engine in ("pyarrow", "fastparquet"):
            try:
                df.to_parquet(tmp, index=False, engine=engine)
                break
            except ImportError:
                continue
        else:
            raise ImportError(
                "Writing parquet requires pyarrow or fastparquet. "
                "Install with: pip install pyarrow"
            )
    else:
        df.to_csv(tmp, index=False)
    tmp.replace(path)


def _iter_table_files(root: Path) -> list[Path]:
    files: list[Path] = []
    tables = root / "tables"
    if tables.is_dir():
        files.extend(sorted(tables.glob("*.parquet")))
        files.extend(sorted(tables.glob("*.csv")))
    for seed_dir in sorted(root.glob("seed_*")):
        seed_tables = seed_dir / "tables"
        if seed_tables.is_dir():
            files.extend(sorted(seed_tables.glob("*.parquet")))
            files.extend(sorted(seed_tables.glob("*.csv")))
    return files


def _row_touches_deleted(row: pd.Series) -> bool:
    for col in MODEL_COLUMNS:
        if col in row.index and row[col] in DELETE_MODELS:
            return True
    return False


def _filter_and_map_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    mask = work.apply(_row_touches_deleted, axis=1)
    work = work.loc[~mask].copy()
    for col in MODEL_COLUMNS:
        if col not in work.columns:
            continue
        work[col] = work[col].map(
            lambda x: OLD_TO_NEW[x] if x in OLD_TO_NEW else x
        )
    return work


def collect_model_names_from_tables(root: Path) -> set[str]:
    names: set[str] = set()
    for fp in _iter_table_files(root):
        df = _read_table(fp)
        for col in MODEL_COLUMNS:
            if col in df.columns:
                names.update(df[col].dropna().astype(str).unique())
    return names


def collect_model_names_from_paths(root: Path) -> set[str]:
    """Best-effort: invert safe names using mapping keys/values."""
    safe_to_logical: dict[str, str] = {}
    for old, new in OLD_TO_NEW.items():
        safe_to_logical[_safe_name(old)] = old
        safe_to_logical[_safe_name(new)] = new
    for deleted in DELETE_MODELS:
        safe_to_logical[_safe_name(deleted)] = deleted

    found: set[str] = set()
    for seed_dir in root.glob("seed_*"):
        baselines = seed_dir / "baselines"
        if baselines.is_dir():
            for child in baselines.iterdir():
                if child.is_dir():
                    found.add(safe_to_logical.get(child.name, f"<path:{child.name}>"))
        attacks = seed_dir / "attacks"
        if attacks.is_dir():
            for fp in attacks.glob("source=*__target=*.npy"):
                m = re.match(r"source=(.+)__target=(.+)\.npy$", fp.name)
                if m:
                    src = safe_to_logical.get(m.group(1), f"<path:{m.group(1)}>")
                    tgt = safe_to_logical.get(m.group(2), f"<path:{m.group(2)}>")
                    found.add(src)
                    found.add(tgt)
        audit = seed_dir / "attacker_audit_adv"
        if audit.is_dir():
            for fp in audit.glob("source=*.npz"):
                m = re.match(r"source=(.+)\.npz$", fp.name)
                if m:
                    found.add(safe_to_logical.get(m.group(1), f"<path:{m.group(1)}>"))
    return found


@dataclass
class DryRunReport:
    unknown_in_tables: set[str] = field(default_factory=set)
    unknown_in_paths: set[str] = field(default_factory=set)
    source_counts: Counter = field(default_factory=Counter)
    target_counts: Counter = field(default_factory=Counter)
    rows_in: int = 0
    rows_out: int = 0
    path_renames: list[tuple[str, str]] = field(default_factory=list)
    paths_to_delete: list[str] = field(default_factory=list)
    chunk_renames: list[tuple[str, str]] = field(default_factory=list)
    collisions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _check_collisions() -> list[str]:
    new_to_old: dict[str, list[str]] = defaultdict(list)
    for old, new in OLD_TO_NEW.items():
        new_to_old[new].append(old)
    return [f"{new} <- {olds}" for new, olds in new_to_old.items() if len(olds) > 1]


def _planned_path_renames(root: Path) -> tuple[list[tuple[str, str]], list[str]]:
    renames: list[tuple[str, str]] = []
    deletions: list[str] = []

    for seed_dir in sorted(root.glob("seed_*")):
        baselines = seed_dir / "baselines"
        if baselines.is_dir():
            for victim_dir in sorted(baselines.iterdir()):
                if not victim_dir.is_dir():
                    continue
                logical = _logical_from_baseline_dir(victim_dir.name)
                if logical in DELETE_MODELS:
                    deletions.append(str(victim_dir))
                    continue
                new_logical = OLD_TO_NEW.get(logical, logical)
                new_dir = baselines / _safe_name(new_logical)
                if victim_dir != new_dir:
                    renames.append((str(victim_dir), str(new_dir)))

        attacks = seed_dir / "attacks"
        if attacks.is_dir():
            for fp in sorted(attacks.glob("source=*__target=*.npy")):
                src_old, tgt_old = _parse_attack_filename(fp.name)
                if src_old in DELETE_MODELS or tgt_old in DELETE_MODELS:
                    deletions.append(str(fp))
                    continue
                src_new = OLD_TO_NEW.get(src_old, src_old)
                tgt_new = OLD_TO_NEW.get(tgt_old, tgt_old)
                new_name = (
                    f"source={_safe_name(src_new)}__target={_safe_name(tgt_new)}.npy"
                )
                new_fp = attacks / new_name
                if fp.name != new_name:
                    renames.append((str(fp), str(new_fp)))

        audit = seed_dir / "attacker_audit_adv"
        if audit.is_dir():
            for fp in sorted(audit.glob("source=*.npz")):
                m = re.match(r"source=(.+)\.npz$", fp.name)
                if not m:
                    continue
                src_old = _logical_from_safe(m.group(1))
                if src_old in DELETE_MODELS:
                    deletions.append(str(fp))
                    continue
                src_new = OLD_TO_NEW.get(src_old, src_old)
                new_fp = audit / f"source={_safe_name(src_new)}.npz"
                if fp != new_fp:
                    renames.append((str(fp), str(new_fp)))

    tables = root / "tables"
    if tables.is_dir():
        for fp in sorted(tables.glob("aggregates_seed=*__source=*.*")):
            m = re.match(
                r"aggregates_seed=(\d+)__source=(.+)\.(parquet|csv)$", fp.name
            )
            if not m:
                continue
            seed, src_safe, ext = m.group(1), m.group(2), m.group(3)
            src_old = _logical_from_safe(src_safe)
            if src_old in DELETE_MODELS:
                deletions.append(str(fp))
                continue
            src_new = OLD_TO_NEW.get(src_old, src_old)
            new_name = f"aggregates_seed={seed}__source={_safe_name(src_new)}.{ext}"
            new_fp = tables / new_name
            if fp.name != new_name:
                renames.append((str(fp), str(new_fp)))

    return renames, deletions


def _logical_from_safe(safe: str) -> str:
    for old in list(OLD_TO_NEW.keys()) + list(DELETE_MODELS):
        if _safe_name(old) == safe:
            return old
    for new in OLD_TO_NEW.values():
        if _safe_name(new) == safe:
            return new
    return f"<unknown_safe:{safe}>"


def _logical_from_baseline_dir(safe: str) -> str:
    return _logical_from_safe(safe)


def _parse_attack_filename(name: str) -> tuple[str, str]:
    m = re.match(r"source=(.+)__target=(.+)\.npy$", name)
    if not m:
        return f"<unknown:{name}>", f"<unknown:{name}>"
    return _logical_from_safe(m.group(1)), _logical_from_safe(m.group(2))


def build_dry_run_report(input_root: Path) -> DryRunReport:
    report = DryRunReport()
    report.collisions = _check_collisions()

    for fp in _iter_table_files(input_root):
        df = _read_table(fp)
        report.rows_in += len(df)
        if "source_model" in df.columns:
            report.source_counts.update(df["source_model"].astype(str))
        if "target_model" in df.columns:
            report.target_counts.update(df["target_model"].astype(str))
        out = _filter_and_map_df(df)
        report.rows_out += len(out)

    table_names = collect_model_names_from_tables(input_root)
    for name in table_names:
        if name in DELETE_MODELS:
            continue
        if name not in OLD_TO_NEW and name not in CANONICAL_NAMES:
            report.unknown_in_tables.add(name)

    path_names = collect_model_names_from_paths(input_root)
    for name in path_names:
        if name.startswith("<"):
            report.unknown_in_paths.add(name)
        elif name in DELETE_MODELS:
            continue
        elif name not in OLD_TO_NEW and name not in CANONICAL_NAMES:
            report.unknown_in_paths.add(name)

    report.path_renames, report.paths_to_delete = _planned_path_renames(input_root)

    dests = [d for _, d in report.path_renames]
    dup_dest = [d for d, c in Counter(dests).items() if c > 1]
    if dup_dest:
        report.warnings.append(f"Duplicate rename targets: {dup_dest[:5]}...")

    return report


def print_dry_run_report(report: DryRunReport, input_root: Path) -> bool:
    ok = True
    print(f"\n=== Dry run: {input_root} ===\n")
    print(f"Rows in tables:  {report.rows_in}")
    print(f"Rows after filter/map: {report.rows_out}")
    print(f"Rows dropped:    {report.rows_in - report.rows_out}")

    if report.collisions:
        print("\nWARNING: multiple old names map to same new name:")
        for line in report.collisions:
            print(f"  {line}")

    if report.unknown_in_tables:
        ok = False
        print("\nUnknown model names in tables:")
        for n in sorted(report.unknown_in_tables):
            print(f"  {n}")

    if report.unknown_in_paths:
        print("\nUnknown / unresolved path names:")
        for n in sorted(report.unknown_in_paths):
            print(f"  {n}")

    print("\nRename map (old -> new), by frequency in source_model:")
    for old, count in report.source_counts.most_common():
        if old in DELETE_MODELS:
            print(f"  DELETE {old} ({count} rows as source)")
        elif old in OLD_TO_NEW:
            print(f"  {old} -> {OLD_TO_NEW[old]} ({count})")
        else:
            print(f"  {old} (unchanged, {count})")

    print(f"\nPlanned path renames: {len(report.path_renames)}")
    for src, dst in report.path_renames[:20]:
        print(f"  {Path(src).name} -> {Path(dst).name}")
    if len(report.path_renames) > 20:
        print(f"  ... and {len(report.path_renames) - 20} more")

    print(f"\nPlanned path deletions: {len(report.paths_to_delete)}")
    for p in report.paths_to_delete[:15]:
        print(f"  {p}")
    if len(report.paths_to_delete) > 15:
        print(f"  ... and {len(report.paths_to_delete) - 15} more")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  {w}")

    print(f"\nDry run {'PASSED' if ok else 'FAILED'} (unknown table names block apply)")
    return ok


def _replace_strings_in_json(obj: Any, replacements: dict[str, str]) -> Any:
    if isinstance(obj, str):
        out = obj
        for old, new in replacements.items():
            if old in out:
                out = out.replace(old, new)
        return out
    if isinstance(obj, dict):
        return {k: _replace_strings_in_json(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_strings_in_json(v, replacements) for v in obj]
    return obj


def _transform_tables(root: Path, report: dict) -> None:
    for fp in tqdm(_iter_table_files(root), desc="Tables"):
        df_in = _read_table(fp)
        n_in = len(df_in)
        df_out = _filter_and_map_df(df_in)
        _write_table(df_out, fp)
        report["tables"].append(
            {"path": str(fp.relative_to(root)), "rows_in": n_in, "rows_out": len(df_out)}
        )


def _rename_with_temp(renames: list[tuple[Path, Path]], report: dict) -> None:
    """Rename files/dirs; use temp names if destination exists."""
    # Sort by path depth descending for directories
    pairs = [(Path(s), Path(d)) for s, d in renames]
    pairs.sort(key=lambda x: len(x[0].parts), reverse=True)

    temp_map: dict[Path, Path] = {}
    for src, dst in pairs:
        if not src.exists():
            report["warnings"].append(f"Missing source for rename: {src}")
            continue
        if src == dst:
            continue
        if dst.exists():
            tmp = src.parent / f".migrate_tmp_{src.name}"
            src.rename(tmp)
            temp_map[tmp] = dst
            report["renames"].append({"from": str(src), "to": str(dst), "via_temp": str(tmp)})
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            report["renames"].append({"from": str(src), "to": str(dst)})

    for tmp, dst in temp_map.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            report["warnings"].append(f"Collision after temp rename: {dst}")
            continue
        tmp.rename(dst)


def _delete_paths(paths: list[str], report: dict) -> None:
    for p in paths:
        path = Path(p)
        if path.is_dir():
            shutil.rmtree(path)
            report["deletions"].append(str(path))
        elif path.is_file():
            path.unlink()
            report["deletions"].append(str(path))


def apply_migration(input_root: Path, output_root: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_root}. "
            "Remove it or change OUTPUT_DIR before applying."
        )

    shutil.copytree(input_root, output_root)
    report: dict = {
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "old_to_new": OLD_TO_NEW,
        "delete_models": sorted(DELETE_MODELS),
        "tables": [],
        "renames": [],
        "deletions": [],
        "warnings": [],
    }

    _transform_tables(output_root, report)

    renames, deletions = _planned_path_renames(output_root)
    _delete_paths(deletions, report)
    _rename_with_temp(renames, report)

    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest = _replace_strings_in_json(manifest, OLD_TO_NEW)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return report


@dataclass
class VerifyResult:
    ok: bool = True
    errors: list[str] = field(default_factory=list)


def _canonical_safe_to_old_logical(safe: str) -> str | None:
    for old, new in OLD_TO_NEW.items():
        if _safe_name(new) == safe:
            return old
    return None


def verify_migration_clean(
    input_root: Path, output_root: Path, migration_report: dict
) -> VerifyResult:
    result = VerifyResult()
    canonical = CANONICAL_NAMES

    expected_rows = sum(t["rows_out"] for t in migration_report.get("tables", []))
    actual_rows = 0
    for fp in _iter_table_files(output_root):
        df = _read_table(fp)
        actual_rows += len(df)
        for col in MODEL_COLUMNS:
            if col not in df.columns:
                continue
            bad = set(df[col].dropna().astype(str)) - canonical
            if bad:
                result.ok = False
                result.errors.append(f"{fp}: non-canonical in {col}: {sorted(bad)}")

    if actual_rows != expected_rows:
        result.ok = False
        result.errors.append(
            f"Row count: expected {expected_rows}, got {actual_rows}"
        )

    for deleted in DELETE_MODELS:
        names = collect_model_names_from_tables(output_root)
        if deleted in names:
            result.ok = False
            result.errors.append(f"Deleted model in output tables: {deleted}")

    for seed_dir in sorted(output_root.glob("seed_*")):
        attacks = seed_dir / "attacks"
        baselines = seed_dir / "baselines"
        if attacks.is_dir() and baselines.is_dir():
            for fp in attacks.glob("source=*__target=*.npy"):
                m = re.match(r"source=(.+)__target=(.+)\.npy$", fp.name)
                if m and not (baselines / m.group(2)).is_dir():
                    result.ok = False
                    result.errors.append(f"No baseline for attack file {fp}")

    for out_seed in sorted(output_root.glob("seed_*")):
        in_seed = input_root / out_seed.name
        if not in_seed.is_dir():
            continue
        out_attacks = out_seed / "attacks"
        in_attacks = in_seed / "attacks"
        if out_attacks.is_dir() and in_attacks.is_dir():
            for out_fp in out_attacks.glob("*.npy"):
                m = re.match(r"source=(.+)__target=(.+)\.npy$", out_fp.name)
                if not m:
                    continue
                out_src, out_tgt = m.group(1), m.group(2)
                in_fp = _find_input_attack_file(in_attacks, out_src, out_tgt)
                if in_fp and in_fp.exists():
                    if np.load(out_fp, mmap_mode="r").shape != np.load(
                        in_fp, mmap_mode="r"
                    ).shape:
                        result.ok = False
                        result.errors.append(
                            f"Shape mismatch {out_fp} vs {in_fp}"
                        )

        out_baselines = out_seed / "baselines"
        in_baselines = in_seed / "baselines"
        if out_baselines.is_dir() and in_baselines.is_dir():
            for out_dir in out_baselines.iterdir():
                if not out_dir.is_dir():
                    continue
                in_dir = _find_input_baseline_dir(in_baselines, out_dir.name)
                if in_dir:
                    for out_f in out_dir.glob("*.npy"):
                        in_f = in_dir / out_f.name
                        if in_f.exists() and np.load(out_f, mmap_mode="r").shape != np.load(
                            in_f, mmap_mode="r"
                        ).shape:
                            result.ok = False
                            result.errors.append(
                                f"Shape mismatch {out_f} vs {in_f}"
                            )

    return result


def _find_input_attack_file(
    in_attacks: Path, out_src_safe: str, out_tgt_safe: str
) -> Path | None:
    src_old = _canonical_safe_to_old_logical(out_src_safe)
    tgt_old = _canonical_safe_to_old_logical(out_tgt_safe)
    if src_old and tgt_old:
        candidate = in_attacks / (
            f"source={_safe_name(src_old)}__target={_safe_name(tgt_old)}.npy"
        )
        if candidate.exists():
            return candidate
    candidate = in_attacks / f"source={out_src_safe}__target={out_tgt_safe}.npy"
    return candidate if candidate.exists() else None


def _find_input_baseline_dir(in_baselines: Path, out_safe: str) -> Path | None:
    old_logical = _canonical_safe_to_old_logical(out_safe)
    if old_logical:
        candidate = in_baselines / _safe_name(old_logical)
        if candidate.is_dir():
            return candidate
    candidate = in_baselines / out_safe
    return candidate if candidate.is_dir() else None


def main() -> None:
    input_root = _resolve(INPUT_DIR)
    output_root = _resolve(OUTPUT_DIR)

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    report = build_dry_run_report(input_root)
    dry_ok = print_dry_run_report(report, input_root)

    if DRY_RUN:
        return

    if not dry_ok:
        raise SystemExit("Dry run failed; fix unknown names before setting DRY_RUN=False")

    migration_report = apply_migration(input_root, output_root)
    migration_report["expected_rows_out"] = report.rows_out

    verify = verify_migration_clean(input_root, output_root, migration_report)
    migration_report["verification_ok"] = verify.ok
    migration_report["verification_errors"] = verify.errors

    report_path = output_root / "migration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(migration_report, f, indent=2)

    print(f"\nWrote migrated tree to: {output_root}")
    print(f"Migration report: {report_path}")

    if verify.ok:
        print("Verification PASSED")
    else:
        print("Verification FAILED:")
        for err in verify.errors:
            print(f"  {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

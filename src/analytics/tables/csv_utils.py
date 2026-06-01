"""CSV merge and clean-accuracy helpers for attack metric tables."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def merge_csv_files(
    file1: str,
    file2: str,
    out: str,
    keys: list[str],
    sort_output: bool = False,
) -> None:
    """Concatenate two CSVs and drop duplicate rows on *keys*."""
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    missing1 = [k for k in keys if k not in df1.columns]
    missing2 = [k for k in keys if k not in df2.columns]
    if missing1 or missing2:
        raise ValueError(
            f"Missing key columns. file1 missing: {missing1}; file2 missing: {missing2}"
        )

    combined = pd.concat([df1, df2], ignore_index=True)
    combined = combined.drop_duplicates(subset=keys, keep="first")

    if sort_output:
        combined = combined.sort_values(keys, kind="stable")

    combined.to_csv(out, index=False)
    log.info("Wrote %s with %s rows.", out, len(combined))


def normalize_model_name(name: str) -> str:
    """Normalize model path strings for matching attack tables to inventory CSVs."""
    if not isinstance(name, str):
        return ""

    name = name.strip().replace("\\", "/")
    name = re.sub(r"\s+", "", name)

    parts = [p for p in name.split("/") if p]
    parts_lower = [p.lower() for p in parts]

    if not parts_lower:
        return ""

    family = parts_lower[0]

    if (
        len(parts_lower) >= 2
        and family == "quantum"
        and parts_lower[1] == "vqm"
    ):
        variant = parts_lower[2] if len(parts_lower) >= 3 else ""
        ansatz_match = re.search(r"ansatz=([a-z_]+)", variant)
        qdepth_match = re.search(r"qdepth=(\d+)", variant)

        if ansatz_match and qdepth_match:
            ansatz = ansatz_match.group(1)
            qdepth = qdepth_match.group(1)

            if qdepth != "50":
                return f"quantum/vqm{qdepth}"

            if ansatz == "basic":
                return f"quantum/vqm{qdepth}-basic"
            if ansatz == "random":
                return f"quantum/vqm{qdepth}-random"
            if ansatz == "strong":
                return f"quantum/vqm{qdepth}-strong"
            if ansatz == "identity":
                return f"quantum/vqm{qdepth}-identity"
            if ansatz == "no_entangling":
                return f"quantum/vqm{qdepth}-noentangling"

        return "/".join(parts_lower)

    return "/".join(parts_lower)


def load_attack_tables(attack_dir: Path) -> pd.DataFrame:
    tables_dir = attack_dir / "tables"
    files = list(tables_dir.glob("*.parquet")) or list(tables_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No attack tables found under {tables_dir}")
    dfs = [
        pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        for f in files
    ]
    return pd.concat(dfs, ignore_index=True)


def extract_clean_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-(seed, target_model) clean accuracy from epsilon=0 rows."""
    work = df.copy()
    work["epsilon"] = pd.to_numeric(work["epsilon"], errors="coerce")
    work["robust_accuracy"] = pd.to_numeric(work["robust_accuracy"], errors="coerce")

    mask = np.isclose(work["epsilon"], 0.0)
    eps0 = work[mask]

    if eps0.empty:
        min_eps = work.groupby(["seed", "target_model"])["epsilon"].min().reset_index()
        eps0 = work.merge(min_eps, on=["seed", "target_model", "epsilon"])

    return (
        eps0.groupby(["seed", "target_model"])["robust_accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"robust_accuracy": "Test Accuracy"})
    )


def append_clean_accuracy_to_tables(
    metrics_csv: Path,
    attack_dir: Path,
    output: Path | None = None,
) -> Path:
    """Merge clean test accuracy from attack tables into an inventory metrics CSV."""
    df = pd.read_csv(metrics_csv)

    def build_model_key(row: pd.Series) -> str:
        parts = [
            str(row["Group / Family"]).strip(),
            str(row["Architecture"]).strip(),
            str(row["Variant Configuration (knobs)"]).strip(),
        ]
        parts = [p for p in parts if p and p.lower() != "nan"]
        return "/".join(parts)

    df["model_key"] = df.apply(build_model_key, axis=1)
    df["model_key"] = df["model_key"].apply(normalize_model_name)

    attack_df = load_attack_tables(attack_dir)
    clean_acc = extract_clean_accuracy(attack_df)
    clean_acc["model_key"] = clean_acc["target_model"].apply(normalize_model_name)

    df["Seed"] = pd.to_numeric(df["Seed"], errors="coerce").astype(int)

    df = df.merge(
        clean_acc[["seed", "model_key", "Test Accuracy"]],
        left_on=["Seed", "model_key"],
        right_on=["seed", "model_key"],
        how="left",
    )
    df = df.drop(columns=["seed"])

    out_path = output or metrics_csv.with_name(metrics_csv.stem + "_with_test_acc.csv")
    df.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)
    return out_path

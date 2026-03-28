from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ============================================================
# Pair metrics builder for attack experiments
#
# Assumptions based on the current logging contract:
# - attack_experiment/tables contains aggregate chunks in parquet or csv
# - each row corresponds to (seed, source_model, target_model, epsilon)
# - expected columns include at least:
#     seed, source_model, target_model, epsilon,
#     robust_accuracy, flip_rate, ASR, transfer_gap
# - optional columns may include:
#     mean_confidence_drop_all / _cc
#     mean_margin_drop_all / _cc
#     mean_true_class_confidence / conf_true / true_class_confidence
#     mean_logit_margin / logit_margin
# - eps=0 is expected in the epsilon grid, allowing clean accuracy to be
#   estimated from robust_accuracy at eps=0 for each target model and seed.
#
# Output:
# - one row per (seed, source_model, target_model)
# - AUC-only metrics, normalized by epsilon range so values stay on the same
#   scale as the original metric whenever possible.
# ============================================================


@dataclass(frozen=True)
class Cols:
    seed: str = "seed"
    source: str = "source_model"
    target: str = "target_model"
    epsilon: str = "epsilon"
    robust_acc: str = "robust_accuracy"
    flip_rate: str = "flip_rate"
    asr: str = "ASR"
    transfer_gap: str = "transfer_gap"


# ---------- configuration ----------

MODEL_BLACKLIST = {
    "Hybrid/hvql/ansatz=basic,qdepth=50",
    "Hybrid/hpvql",
    "Quantum/VQM50-Identity",
}

# ---------- helpers ----------

def _find_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalized_auc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Normalized trapezoidal AUC over x, returning a value on the scale of y.
    Returns NaN if fewer than 2 valid points exist.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return float("nan")
    if x.size == 1:
        return float(y[0])

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    width = x[-1] - x[0]
    if width <= 0:
        return float(np.mean(y))

    return float(np.trapezoid(y, x) / width)


def _safe_div(num: float, den: float) -> float:
    if den is None or not np.isfinite(den) or den == 0:
        return float("nan")
    return float(num / den)


def _parse_family_arch_variant(model_name: str) -> tuple[Optional[str], Optional[str], str]:
    """
    Best-effort parser for names like:
      Classical/convnet/conv_channels=[8],fc_sizes=[32]
      Hybrid/hvql/ansatz=basic,qdepth=50
      VQM50-Strong
      hvql

    This is intentionally conservative. You can replace this parser if your
    canonical model naming scheme is known.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        return None, None, model_name

    raw = model_name.strip().replace("\\", "/")
    parts = [p for p in raw.split("/") if p]

    family = None
    arch = None
    variant = raw

    families = {"classical": "Classical", "hybrid": "Hybrid", "quantum": "Quantum"}

    if parts:
        p0 = parts[0].lower()
        if p0 in families:
            family = families[p0]
            if len(parts) >= 2:
                arch = parts[1]
            if len(parts) >= 3:
                variant = parts[2]
            elif len(parts) >= 2:
                variant = parts[1]
            return family, arch, variant

    lower = raw.lower()
    if lower.startswith("vqm"):
        family = "Quantum"
        arch = raw.split("-")[0]
        variant = raw
    elif lower.startswith(("hv", "hpv", "hqe", "hquanv", "drophvql", "reshvql")):
        family = "Hybrid"
        arch = raw.split("/")[0].split("-")[0]
        variant = raw
    elif lower.startswith(("mlp", "convnet", "lenet", "resnet", "mlpmixer")):
        family = "Classical"
        arch = raw.split("/")[0].split("-")[0]
        variant = raw

    return family, arch, variant


def _read_attack_tables(attack_dir: Path) -> pd.DataFrame:
    tables_dir = attack_dir / "tables"
    if not tables_dir.exists():
        raise FileNotFoundError(f"Could not find tables directory: {tables_dir}")

    parquet_files = sorted(tables_dir.glob("*.parquet"))
    csv_files = sorted(tables_dir.glob("*.csv"))

    frames: list[pd.DataFrame] = []
    if parquet_files:
        for fp in parquet_files:
            frames.append(pd.read_parquet(fp))
    elif csv_files:
        for fp in csv_files:
            frames.append(pd.read_csv(fp))
    else:
        raise FileNotFoundError(f"No parquet/csv attack tables found in {tables_dir}")

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError("Attack aggregate table is empty.")
    return df


def _infer_clean_accuracy(df: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    """
    Clean accuracy per (seed, target_model), inferred from robust_accuracy at eps=0.
    Falls back to minimum epsilon if exact 0.0 is absent.
    """
    work = df[[cols.seed, cols.target, cols.epsilon, cols.robust_acc]].copy()
    work[cols.epsilon] = pd.to_numeric(work[cols.epsilon], errors="coerce")

    eps0_mask = np.isclose(work[cols.epsilon].to_numpy(dtype=float), 0.0, atol=1e-12)
    eps0 = work.loc[eps0_mask].copy()

    if eps0.empty:
        min_eps = work.groupby([cols.seed, cols.target], as_index=False)[cols.epsilon].min()
        eps0 = work.merge(min_eps, on=[cols.seed, cols.target, cols.epsilon], how="inner")

    clean = (
        eps0.groupby([cols.seed, cols.target], as_index=False)[cols.robust_acc]
        .mean()
        .rename(columns={cols.robust_acc: "clean_accuracy"})
    )
    return clean


def build_pair_metrics(
    attack_dir: Path,
    output_path: Path,
    eps_min: Optional[float] = None,
    eps_max: Optional[float] = None,
    model_blacklist: Optional[set[str]] = None,
) -> pd.DataFrame:
    cols = Cols()
    df = _read_attack_tables(attack_dir)

    required = [cols.seed, cols.source, cols.target, cols.epsilon, cols.robust_acc, cols.flip_rate, cols.asr]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in attack aggregates: {missing}")

    df = df.copy()
    df[cols.epsilon] = pd.to_numeric(df[cols.epsilon], errors="coerce")
    df[cols.robust_acc] = pd.to_numeric(df[cols.robust_acc], errors="coerce")
    df[cols.flip_rate] = pd.to_numeric(df[cols.flip_rate], errors="coerce")
    df[cols.asr] = pd.to_numeric(df[cols.asr], errors="coerce")
    if cols.transfer_gap in df.columns:
        df[cols.transfer_gap] = pd.to_numeric(df[cols.transfer_gap], errors="coerce")

    if eps_min is not None:
        df = df[df[cols.epsilon] >= eps_min]
    if eps_max is not None:
        df = df[df[cols.epsilon] <= eps_max]

    if df.empty:
        raise ValueError("No rows left after epsilon filtering.")

    blacklist = {m.strip() for m in (model_blacklist or set()) if isinstance(m, str) and m.strip()}
    if blacklist:
        before_rows = len(df)
        df = df[~df[cols.source].astype(str).isin(blacklist)]
        df = df[~df[cols.target].astype(str).isin(blacklist)]
        after_rows = len(df)
        removed_rows = before_rows - after_rows
        print(f"Blacklisted rows removed: {removed_rows}")

    if df.empty:
        raise ValueError("No rows left after blacklist filtering.")

    conf_true_col = _find_first_existing(
        df,
        [
            "true_class_confidence",
            "mean_true_class_confidence",
            "conf_true",
            "conf_true_all",
            "conf_true_cc",
        ],
    )
    margin_col = _find_first_existing(
        df,
        [
            "logit_margin",
            "mean_logit_margin",
            "margin",
            "margin_all",
            "margin_cc",
        ],
    )
    conf_drop_col = _find_first_existing(
        df,
        [
            "mean_confidence_drop_all",
            "mean_confidence_drop_cc",
            "confidence_drop",
            "conf_drop",
        ],
    )
    margin_drop_col = _find_first_existing(
        df,
        [
            "mean_margin_drop_all",
            "mean_margin_drop_cc",
            "margin_drop",
        ],
    )

    clean_df = _infer_clean_accuracy(df, cols)
    df = df.merge(clean_df, on=[cols.seed, cols.target], how="left")

    def _agg_pair(seed_value: object, source_model: str, target_model: str, group: pd.DataFrame) -> pd.Series:
        x = group[cols.epsilon].to_numpy(dtype=float)
        robust = group[cols.robust_acc].to_numpy(dtype=float)
        flip = group[cols.flip_rate].to_numpy(dtype=float)
        asr = group[cols.asr].to_numpy(dtype=float)
        clean_acc = float(group["clean_accuracy"].dropna().iloc[0]) if group["clean_accuracy"].notna().any() else float("nan")

        robust_auc = _normalized_auc(x, robust)
        flip_auc = _normalized_auc(x, flip)
        asr_auc = _normalized_auc(x, asr)
        acc_drop_auc = clean_acc - robust_auc if np.isfinite(clean_acc) and np.isfinite(robust_auc) else float("nan")
        retention_auc = _safe_div(robust_auc, clean_acc)

        transfer_gap_auc = float("nan")
        if cols.transfer_gap in group.columns:
            transfer_gap_auc = _normalized_auc(x, group[cols.transfer_gap].to_numpy(dtype=float))

        conf_true_auc = float("nan")
        if conf_true_col is not None:
            conf_true_auc = _normalized_auc(x, pd.to_numeric(group[conf_true_col], errors="coerce").to_numpy(dtype=float))

        conf_drop_auc = float("nan")
        if conf_drop_col is not None:
            conf_drop_auc = _normalized_auc(x, pd.to_numeric(group[conf_drop_col], errors="coerce").to_numpy(dtype=float))

        margin_auc = float("nan")
        if margin_col is not None:
            margin_auc = _normalized_auc(x, pd.to_numeric(group[margin_col], errors="coerce").to_numpy(dtype=float))

        margin_drop_auc = float("nan")
        if margin_drop_col is not None:
            margin_drop_auc = _normalized_auc(x, pd.to_numeric(group[margin_drop_col], errors="coerce").to_numpy(dtype=float))

        is_self = source_model == target_model
        flag = "WB" if is_self else "BB"

        src_family, src_arch, src_variant = _parse_family_arch_variant(source_model)
        tgt_family, tgt_arch, tgt_variant = _parse_family_arch_variant(target_model)

        transfer_robust_auc = robust_auc if not is_self else float("nan")
        transfer_asr_auc = asr_auc if not is_self else float("nan")
        transfer_gap_auc_out = transfer_gap_auc if not is_self else float("nan")

        return pd.Series(
            {
                "Seed": seed_value,
                "Source Model": source_model,
                "Source Family": src_family,
                "Source Architecture": src_arch,
                "Source Variant": src_variant,
                "Target Model": target_model,
                "Target Family": tgt_family,
                "Target Architecture": tgt_arch,
                "Target Variant": tgt_variant,
                "WB/BB Flag": flag,
                "Is Self Attack": is_self,
                "Clean Accuracy": clean_acc,
                "Robust Accuracy": robust_auc,
                "Accuracy Drop": acc_drop_auc,
                "Retention": retention_auc,
                "Flip Rate": flip_auc,
                "ASR": asr_auc,
                "Transfer Robust Accuracy": transfer_robust_auc,
                "Transfer ASR": transfer_asr_auc,
                "Transfer Gap": transfer_gap_auc_out,
                "True-class Confidence": conf_true_auc,
                "Mean Confidence Drop": conf_drop_auc,
                "Logit Margin": margin_auc,
                "Mean Margin Drop": margin_drop_auc,
            }
        )

    rows: list[pd.Series] = []
    for (seed_value, source_model, target_model), group in df.groupby(
        [cols.seed, cols.source, cols.target], sort=False, dropna=False
    ):
        rows.append(_agg_pair(seed_value, str(source_model), str(target_model), group))

    per_seed_df = pd.DataFrame(rows)

    per_seed_df = per_seed_df.sort_values(["Seed", "Source Family", "Source Architecture", "Source Model", "Target Family", "Target Architecture", "Target Model"]).reset_index(drop=True)

    # ==========================
    # Aggregate over seeds
    # ==========================

    agg_cols = [
        "Clean Accuracy",
        "Robust Accuracy",
        "Accuracy Drop",
        "Retention",
        "Flip Rate",
        "ASR",
        "Transfer Robust Accuracy",
        "Transfer ASR",
        "Transfer Gap",
        "True-class Confidence",
        "Mean Confidence Drop",
        "Logit Margin",
        "Mean Margin Drop",
    ]

    group_cols = [
        "Source Model",
        "Source Family",
        "Source Architecture",
        "Source Variant",
        "Target Model",
        "Target Family",
        "Target Architecture",
        "Target Variant",
        "WB/BB Flag",
        "Is Self Attack",
    ]

    pair_df = (
        per_seed_df.groupby(group_cols, dropna=False)[agg_cols]
        .mean()
        .reset_index()
    )

    # Add seed count
    seed_counts = (
        per_seed_df.groupby(group_cols, dropna=False)["Seed"]
        .nunique()
        .reset_index()
        .rename(columns={"Seed": "#Seeds Aggregated"})
    )

    pair_df = pair_df.merge(seed_counts, on=group_cols, how="left")

    # ==========================
    # Save both tables
    # ==========================

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        pair_df.to_parquet(output_path, index=False)
        per_seed_path = output_path.with_name(f"{output_path.stem}_per_seed.parquet")
        per_seed_df.to_parquet(per_seed_path, index=False)
    else:
        pair_df.to_csv(output_path, index=False)
        per_seed_path = output_path.with_name(f"{output_path.stem}_per_seed.csv")
        per_seed_df.to_csv(per_seed_path, index=False)

    return pair_df, per_seed_df


def main() -> None:
    attack_dir = Path("logs/ATTACKS")
    output = Path("outputs/pair_metrics.csv")
    eps_min = None
    eps_max = None

    if attack_dir.name != "attack_experiment":
        maybe = attack_dir / "attack_experiment"
        if maybe.exists():
            attack_dir = maybe

    pair_df, per_seed_df = build_pair_metrics(
        attack_dir=attack_dir,
        output_path=output,
        eps_min=eps_min,
        eps_max=eps_max,
        model_blacklist=MODEL_BLACKLIST,
    )

    per_seed_output = output.with_name(f"{output.stem}_per_seed{output.suffix}")

    print(f"Saved pair metrics: {output}")
    print(f"Saved per-seed debug metrics: {per_seed_output}")
    print(f"Rows (aggregated): {len(pair_df)}")
    print(f"Rows (per-seed): {len(per_seed_df)}")
    print(f"Columns (aggregated): {len(pair_df.columns)}")

    wb = int((pair_df["WB/BB Flag"] == "WB").sum())
    bb = int((pair_df["WB/BB Flag"] == "BB").sum())
    print(f"WB rows: {wb}")
    print(f"BB rows: {bb}")

    if MODEL_BLACKLIST:
        print(f"Blacklisted models: {sorted(MODEL_BLACKLIST)}")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from pathlib import Path
import re


# =========================
# CONFIG (EDIT THIS)
# =========================
CSV_PATH = Path("metrics.csv")
ATTACK_DIR = Path("logs/ATTACKS/attack_experiment")  # update this


# ================= NORMALIZATION =================
def normalize_model_name(name: str) -> str:
    if not isinstance(name, str):
        return ""

    name = name.strip().replace("\\", "/")
    name = re.sub(r"\s+", "", name)

    parts = [p for p in name.split("/") if p]
    parts_lower = [p.lower() for p in parts]

    # normalize family
    if parts_lower:
        family = parts_lower[0]
    else:
        return ""

    # ----- Quantum special case ONLY -----
    # Attack names:
    #   Quantum/VQM50-Basic
    #   Quantum/VQM100
    # CSV names:
    #   Quantum/VQM/ansatz=basic,qdepth=50
    #   Quantum/VQM/ansatz=strong,qdepth=100
    #
    # IMPORTANT: do not trigger on HVQM / HPVQM.
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

            # map back to attack-style naming
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

    # ----- everything else: just normalize casing/spaces -----
    return "/".join(parts_lower)

# ================= LOAD ATTACK TABLES =================
def load_attack_tables(attack_dir: Path):
    tables_dir = attack_dir / "tables"
    files = list(tables_dir.glob("*.parquet")) or list(tables_dir.glob("*.csv"))
    dfs = [pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


# ================= EXTRACT CLEAN ACC =================
def extract_clean_accuracy(df):
    df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
    df["robust_accuracy"] = pd.to_numeric(df["robust_accuracy"], errors="coerce")

    mask = np.isclose(df["epsilon"], 0.0)
    eps0 = df[mask]

    if eps0.empty:
        min_eps = df.groupby(["seed", "target_model"])["epsilon"].min().reset_index()
        eps0 = df.merge(min_eps, on=["seed", "target_model", "epsilon"])

    out = (
        eps0.groupby(["seed", "target_model"])["robust_accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"robust_accuracy": "Test Accuracy"})
    )

    return out


# ================= MAIN =================
def main():
    df = pd.read_csv(CSV_PATH)

    # Build model column
    def build_model_key(row):
        parts = [
            str(row["Group / Family"]).strip(),
            str(row["Architecture"]).strip(),
            str(row["Variant Configuration (knobs)"]).strip(),
        ]

        # keep only non-empty parts
        parts = [p for p in parts if p and p.lower() != "nan"]

        return "/".join(parts)

    df["model_key"] = df.apply(build_model_key, axis=1)

    df["model_key"] = df["model_key"].apply(normalize_model_name)

    attack_df = load_attack_tables(ATTACK_DIR)
    clean_acc = extract_clean_accuracy(attack_df)

    clean_acc["model_key"] = clean_acc["target_model"].apply(normalize_model_name)

    df["Seed"] = pd.to_numeric(df["Seed"], errors="coerce").astype(int)

    df = df.merge(
        clean_acc[["seed", "model_key", "Test Accuracy"]],
        left_on=["Seed", "model_key"],
        right_on=["seed", "model_key"],
        how="left"
    )

    df = df.drop(columns=["seed"])

    output = CSV_PATH.with_name(CSV_PATH.stem + "_with_test_acc.csv")
    df.to_csv(output, index=False)

    print("Done:", output)


if __name__ == "__main__":
    main()
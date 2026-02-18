from __future__ import annotations

"""
Baseline training summary table (simplified).

Design goals:
- One function per extracted metric (easy to read / modify).
- Reuse your MetricsContainer for metrics loading + epoch-aligned DataFrame.
- Minimal configuration: just paths in, CSV + LaTeX out.

Expected run_dir layout (seed directory):
- metrics/logger.pkl  (written by MetricsContainer.save_metrics)
- final.ckpt          (optional, for parameter counting)
- model.info          (optional, for time/hardware)
- training.log        (optional, fallback for time/hardware)
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch

from metrics import MetricsContainer


# ---------------------------
# Small helpers
# ---------------------------

@dataclass(frozen=True)
class Run:
    path: Path

    def metrics_df(self) -> pd.DataFrame:
        """Load metrics via MetricsContainer -> pandas, or return empty df."""
        container = MetricsContainer(str(self.path))
        try:
            container.load_metrics(str(self.path))
            return container.to_pandas()
        except Exception:
            return pd.DataFrame()

    def read_text(self, filename: str) -> str | None:
        p = self.path / filename
        if not p.exists():
            return None
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

    def load_state_dict(self) -> dict[str, Any] | None:
        ckpt = self.path / "final.ckpt"
        if not ckpt.exists():
            return None
        try:
            payload = torch.load(ckpt, map_location="cpu")
        except Exception:
            return None

        if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
            return payload["state_dict"]
        if isinstance(payload, dict):
            return payload
        return None


def _last_numeric(df: pd.DataFrame, candidates: Iterable[str]) -> float | None:
    for col in candidates:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not s.empty:
            return float(s.iloc[-1])
    return None


def _extract_seconds(text: str) -> float | None:
    # "Total training time : 123.45 seconds"
    m = re.search(
        r"Total training time\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return float(m.group(1))

    # "HH:MM:SS(.sss)"
    m = re.search(r"(\d+):(\d{2}):(\d{2}(?:\.\d+)?)", text)
    if m:
        h, mm, ss = m.groups()
        return int(h) * 3600 + int(mm) * 60 + float(ss)

    return None


def _format_hms(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------
# One function per metric
# ---------------------------

def extract_train_final_acc(run_dir: str | Path) -> float | None:
    run = Run(Path(run_dir))
    df = run.metrics_df()
    if df.empty:
        return None
    # MetricsContainer.to_pandas prefixes group names with "train_" / "val_".
    return _last_numeric(df, ["train_acc", "train/acc", "train_accuracy", "acc"])


def extract_val_final_acc(run_dir: str | Path) -> float | None:
    run = Run(Path(run_dir))
    df = run.metrics_df()
    if df.empty:
        return None
    return _last_numeric(df, ["val_acc", "val/acc", "valid_acc", "validation_acc"])


def extract_training_time_seconds(run_dir: str | Path) -> float | None:
    run = Run(Path(run_dir))

    info = run.read_text("model.info")
    if info:
        v = _extract_seconds(info)
        if v is not None:
            return v

    log_text = run.read_text("training.log")
    if log_text:
        v = _extract_seconds(log_text)
        if v is not None:
            return v

    return None


def extract_hardware(run_dir: str | Path) -> str | None:
    run = Run(Path(run_dir))

    info = run.read_text("model.info")
    if info:
        for line in info.splitlines():
            if "hardware" in line.lower():
                return line.split(":", 1)[-1].strip() if ":" in line else line.strip()

    log_text = run.read_text("training.log")
    if log_text:
        for line in log_text.splitlines():
            lowered = line.lower()
            if "available" in lowered and any(tok in lowered for tok in ("gpu", "cuda", "cpu", "tpu", "mps")):
                return line.strip()

    return None


def extract_num_params(run_dir: str | Path) -> int | None:
    run = Run(Path(run_dir))
    sd = run.load_state_dict()
    if not sd:
        return None
    return int(sum(t.numel() for t in sd.values() if hasattr(t, "numel")))


def extract_num_quantum_params(run_dir: str | Path, qname_pattern: str = "qlayer") -> int | None:
    run = Run(Path(run_dir))
    sd = run.load_state_dict()
    if not sd:
        return None

    pat = re.compile(qname_pattern)
    total = 0
    for name, tensor in sd.items():
        if pat.search(name) and hasattr(tensor, "numel"):
            total += int(tensor.numel())
    return total


def extract_seed(run_dir: str | Path) -> int | None:
    p = Path(run_dir)

    m = re.search(r"seed[_=-]?(\d+)", p.name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    if p.name.isdigit():
        return int(p.name)

    for part in reversed(p.parts):
        if part.isdigit():
            return int(part)

    return None


def extract_model_label(run_dir: str | Path) -> str:
    p = Path(run_dir)
    if p.name.lower().startswith("seed") and p.parent.name:
        return p.parent.name
    return p.name


# ---------------------------
# Table generation
# ---------------------------

def build_summary_dataframe(run_dirs: list[str | Path], *, qname_pattern: str = "qlayer") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rd in run_dirs:
        run_dir = Path(rd).expanduser().resolve()
        rows.append(
            {
                "Model label": extract_model_label(run_dir),
                "Seed": extract_seed(run_dir),
                "Train final accuracy": extract_train_final_acc(run_dir),
                "Val final accuracy": extract_val_final_acc(run_dir),
                "Training time": _format_hms(extract_training_time_seconds(run_dir)),
                "Hardware": extract_hardware(run_dir) or "--",
                "parameters": extract_num_params(run_dir),
                "quantum parameters": extract_num_quantum_params(run_dir, qname_pattern=qname_pattern),
                "Test accuracy": "TODO",
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    latex_filename: str = "baseline_training_summary.tex",
    full_table_env: bool = False,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "baseline_training_summary.csv"
    df.to_csv(csv_path, index=False)

    latex_path = out_dir / latex_filename
    tabular = df.to_latex(index=False, escape=True, column_format="llccclrrc")

    latex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"{tabular}\n"
        "\\caption{Baseline training summary.}\n"
        "\\label{tab:baseline_training_summary}\n"
        "\\end{table}\n"
        if full_table_env
        else tabular
    )
    latex_path.write_text(latex, encoding="utf-8")
    return csv_path, latex_path


def generate_baseline_training_summary(
    run_dirs: list[str | Path],
    *,
    out_dir: str | Path = "outputs/tables",
    qname_pattern: str = "qlayer",
    latex_filename: str = "baseline_training_summary.tex",
    full_table_env: bool = False,
) -> tuple[Path, Path]:
    df = build_summary_dataframe(run_dirs, qname_pattern=qname_pattern)
    return write_outputs(df, out_dir, latex_filename=latex_filename, full_table_env=full_table_env)


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate baseline training summary table from run directories.")
    p.add_argument("--runs", action="append", required=True, help="Run directory path. Repeat for multiple runs.")
    p.add_argument("--qname_pattern", default="qlayer", help="Regex pattern to identify quantum parameter names.")
    p.add_argument("--out_dir", default="outputs/tables", help="Output directory for CSV and LaTeX files.")
    p.add_argument("--latex", default="baseline_training_summary.tex", help="Output LaTeX filename.")
    p.add_argument("--full_table_env", action="store_true", help="Wrap output in a full LaTeX table environment.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path, latex_path = generate_baseline_training_summary(
        args.runs,
        out_dir=args.out_dir,
        qname_pattern=args.qname_pattern,
        latex_filename=args.latex,
        full_table_env=args.full_table_env,
    )
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote LaTeX: {latex_path}")


if __name__ == "__main__":
    main()

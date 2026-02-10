from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def _find_metrics_csv(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "metrics" / "metrics.csv",
        run_dir / "metrics.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_logger_pkl(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "metrics" / "logger.pkl",
        run_dir / "logger.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_metrics_df(run_dir: Path) -> pd.DataFrame:
    metrics_csv = _find_metrics_csv(run_dir)
    if metrics_csv is not None:
        try:
            return pd.read_csv(metrics_csv)
        except Exception:
            pass

    logger_pkl = _find_logger_pkl(run_dir)
    if logger_pkl is not None:
        try:
            with open(logger_pkl, "rb") as fp:
                payload = pickle.load(fp)
            raw = payload.get("_raw", {}) if isinstance(payload, dict) else {}
            frames: list[pd.DataFrame] = []
            epoch_steps = raw.get("epoch", [])
            if not epoch_steps:
                return pd.DataFrame()

            def _assign_epochs(steps: pd.Series) -> list[int]:
                epochs = []
                boundary_idx = 0
                n_epochs = len(epoch_steps)
                for step in steps:
                    while boundary_idx < n_epochs and step > epoch_steps[boundary_idx]:
                        boundary_idx += 1
                    epochs.append(boundary_idx)
                return epochs

            for group, metrics in raw.items():
                if group == "epoch" or not isinstance(metrics, dict):
                    continue
                frame = pd.DataFrame(metrics)
                if frame.empty or "step" not in frame.columns:
                    continue
                frame = frame.sort_values("step").reset_index(drop=True)
                frame.insert(0, "epoch", _assign_epochs(frame["step"]))
                frame = frame.rename(
                    columns={
                        col: (
                            col
                            if col in {"epoch", "step"} and group == "train"
                            else (f"{group}_{col}" if col != "epoch" else "epoch")
                        )
                        for col in frame.columns
                    }
                )
                frames.append(frame)
            if not frames:
                return pd.DataFrame()
            merged = frames[0]
            for frame in frames[1:]:
                merged = pd.merge(merged, frame, on="epoch", how="outer", sort=True)
            return merged.sort_values("epoch").reset_index(drop=True)
        except Exception:
            pass

    return pd.DataFrame()


def _extract_last_metric(run_dir: Path, candidates: list[str]) -> float | None:
    df = _load_metrics_df(run_dir)
    if df.empty:
        return None

    for column in candidates:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if not series.empty:
            return float(series.iloc[-1])

    return None


def _read_model_info(run_dir: Path) -> str | None:
    model_info_path = run_dir / "model.info"
    if not model_info_path.exists():
        return None
    try:
        return model_info_path.read_text(encoding="utf-8")
    except Exception:
        return None


def _extract_seconds_from_text(text: str) -> float | None:
    match = re.search(r"Total training time\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))

    match = re.search(r"(\d+):(\d{2}):(\d{2}(?:\.\d+)?)", text)
    if match:
        h, m, s = match.groups()
        return int(h) * 3600 + int(m) * 60 + float(s)

    return None


def extract_train_final_acc(run_dir: str | Path) -> float | None:
    run_dir = Path(run_dir)
    return _extract_last_metric(run_dir, ["train_acc", "train/acc", "acc", "train_accuracy"])


def extract_val_final_acc(run_dir: str | Path) -> float | None:
    run_dir = Path(run_dir)
    return _extract_last_metric(run_dir, ["val_acc", "val/acc", "valid_acc", "validation_acc"])


def extract_training_time(run_dir: str | Path) -> float | None:
    run_dir = Path(run_dir)

    model_info = _read_model_info(run_dir)
    if model_info:
        duration = _extract_seconds_from_text(model_info)
        if duration is not None:
            return duration

    train_log = run_dir / "training.log"
    if train_log.exists():
        try:
            log_text = train_log.read_text(encoding="utf-8", errors="ignore")
            duration = _extract_seconds_from_text(log_text)
            if duration is not None:
                return duration
        except Exception:
            pass

    return None


def extract_hardware(run_dir: str | Path) -> str | None:
    run_dir = Path(run_dir)

    model_info = _read_model_info(run_dir)
    if model_info:
        for line in model_info.splitlines():
            if "hardware" in line.lower():
                return line.split(":", 1)[-1].strip() if ":" in line else line.strip()

    train_log = run_dir / "training.log"
    if train_log.exists():
        try:
            for line in train_log.read_text(encoding="utf-8", errors="ignore").splitlines():
                lowered = line.lower()
                if any(token in lowered for token in ["gpu", "cuda", "cpu", "tpu", "mps"]) and "available" in lowered:
                    return line.strip()
        except Exception:
            pass

    return None


def _extract_total_params_from_model_info(model_info: str) -> int | None:
    match = re.search(r"Total params:\s*([0-9,]+)", model_info)
    if match:
        return int(match.group(1).replace(",", ""))
    return None


def _load_checkpoint_state_dict(run_dir: Path) -> dict[str, Any] | None:
    ckpt_path = run_dir / "final.ckpt"
    if not ckpt_path.exists():
        return None

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt if isinstance(ckpt, dict) else None)
        if isinstance(state_dict, dict):
            return state_dict
    except Exception:
        return None

    return None


def extract_num_params(run_dir: str | Path) -> int | None:
    run_dir = Path(run_dir)

    state_dict = _load_checkpoint_state_dict(run_dir)
    if state_dict:
        return int(sum(v.numel() for v in state_dict.values() if hasattr(v, "numel")))

    model_info = _read_model_info(run_dir)
    if model_info:
        return _extract_total_params_from_model_info(model_info)

    return None


def extract_num_quantum_params(run_dir: str | Path, qname_pattern: str = "qlayer") -> int | None:
    run_dir = Path(run_dir)

    state_dict = _load_checkpoint_state_dict(run_dir)
    if not state_dict:
        return None

    pattern = re.compile(qname_pattern)
    quantum_count = 0
    for name, tensor in state_dict.items():
        if pattern.search(name) and hasattr(tensor, "numel"):
            quantum_count += int(tensor.numel())

    return quantum_count


def extract_seed(run_dir: str | Path) -> int | None:
    run_dir = Path(run_dir)

    match = re.search(r"seed[_=-]?(\d+)", run_dir.name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    if run_dir.name.isdigit():
        return int(run_dir.name)

    for part in reversed(run_dir.parts):
        if part.isdigit():
            return int(part)

    return None


def extract_model_label(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    if run_dir.name.lower().startswith("seed") and run_dir.parent.name:
        return run_dir.parent.name
    return run_dir.name


def _format_training_time(value: float | None) -> str:
    if value is None:
        return "--"
    total_seconds = int(round(value))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_summary_dataframe(models: list[str], qname_pattern: str = "qlayer") -> pd.DataFrame:
    rows = []
    for model_dir in models:
        run_dir = Path(model_dir)
        rows.append(
            {
                "Model label": extract_model_label(run_dir),
                "Seed": extract_seed(run_dir),
                "Train final accuracy": extract_train_final_acc(run_dir),
                "Val final accuracy": extract_val_final_acc(run_dir),
                "Training time": _format_training_time(extract_training_time(run_dir)),
                "Hardware": extract_hardware(run_dir) or "--",
                "parameters": extract_num_params(run_dir),
                "quantum parameters": extract_num_quantum_params(run_dir, qname_pattern=qname_pattern),
                # TODO: join with attack aggregates to fetch clean test acc once available
                "Test accuracy": "TODO",
            }
        )

    return pd.DataFrame(rows)


def write_outputs(
    df: pd.DataFrame,
    out_dir: str | Path,
    latex_filename: str,
    full_table_env: bool = False,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "baseline_training_summary.csv"
    df.to_csv(csv_path, index=False)

    latex_path = out_dir / latex_filename
    latex_tabular = df.to_latex(index=False, escape=True, column_format="llccclrrc")

    if full_table_env:
        latex_content = (
            "\\begin{table}[t]\n"
            "\\centering\n"
            f"{latex_tabular}\n"
            "\\caption{Baseline training summary.}\n"
            "\\label{tab:baseline_training_summary}\n"
            "\\end{table}\n"
        )
    else:
        latex_content = latex_tabular

    latex_path.write_text(latex_content, encoding="utf-8")
    return csv_path, latex_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline training summary table from model run directories.")
    parser.add_argument("--models", action="append", required=True, help="Model run directory path. Repeat for multiple models.")
    parser.add_argument("--qname_pattern", default="qlayer", help="Regex pattern to identify quantum parameter names.")
    parser.add_argument("--out_dir", default="outputs/tables", help="Output directory for CSV and LaTeX files.")
    parser.add_argument("--latex", default="baseline_training_summary.tex", help="Output LaTeX filename.")
    parser.add_argument("--full_table_env", action="store_true", help="Wrap output in a full LaTeX table environment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_summary_dataframe(models=args.models, qname_pattern=args.qname_pattern)
    csv_path, latex_path = write_outputs(df=df, out_dir=args.out_dir, latex_filename=args.latex, full_table_env=args.full_table_env)
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote LaTeX: {latex_path}")


if __name__ == "__main__":
    main()

import re
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO

from hydra.utils import log

def parse_profiler_txt(path: str) -> pd.DataFrame:
    """
    Parse a PyTorch Lightning or cProfile-like profiler text output.
    Returns a DataFrame with section, function, and timing metrics.
    """
    text = Path(path).read_text(errors="ignore")
    number_row = re.compile(r"^\s*(\d+)(?:/(\d+))?\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(.+)$")
    rows = []
    lines = text.splitlines()
    i = 0
    current_section = None
    while i < len(lines):
        line = lines[i]
        if line.startswith("Profile stats for:"):
            current_section = line.split("Profile stats for:", 1)[1].strip()
            i += 1
            while i < len(lines) and "ncalls" not in lines[i]:
                i += 1
            i += 1
            while i < len(lines):
                row = lines[i]
                if not row.strip():
                    break
                m = number_row.match(row)
                if m:
                    total = int(m.group(1))
                    prim = int(m.group(2)) if m.group(2) else total
                    tottime = float(m.group(3))
                    percall_self = float(m.group(4))
                    cumtime = float(m.group(5))
                    percall_cum = float(m.group(6))
                    fn = m.group(7).strip()
                    rows.append({
                        "section": current_section,
                        "ncalls_total": total,
                        "ncalls_primitive": prim,
                        "tottime_s": tottime,
                        "percall_self_s": percall_self,
                        "cumtime_s": cumtime,
                        "percall_cum_s": percall_cum,
                        "function": fn
                    })
                i += 1
        else:
            i += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No rows parsed from {path}. Check file format.")
    section_totals = df.groupby("section", dropna=False)["cumtime_s"].sum().reset_index(name="section_cumtime_s")
    df = df.merge(section_totals, on="section", how="left")
    df["pct_of_section"] = (df["cumtime_s"] / df["section_cumtime_s"]) * 100.0
    return df


def print_filtered_summary(
    df: pd.DataFrame,
    output_path: str = "summary-profile.txt",
    section_time_thresh: float = 0.1,
    section_share_thresh: float = None,
    top_k_per_section: int = 3,
    top_n_sections: int = 5,
):
    """
    Filter sections and write a concise summary to a text file.
    """
    buf = StringIO()
    total_time = df["cumtime_s"].sum()
    section_totals = df.groupby("section", dropna=False)["section_cumtime_s"].first().reset_index()
    section_totals = section_totals.sort_values("section_cumtime_s", ascending=False)

    mask = section_totals["section_cumtime_s"] >= section_time_thresh
    if section_share_thresh is not None:
        section_totals["section_share_pct"] = (section_totals["section_cumtime_s"] / total_time) * 100.0
        mask &= section_totals["section_share_pct"] >= section_share_thresh

    filtered_sections = section_totals[mask].head(top_n_sections)
    if filtered_sections.empty:
        buf.write("No sections pass the threshold filters.\n")
    else:
        buf.write(f"=== Top {len(filtered_sections)} sections by total time (after filters) ===\n")
        for _, row in filtered_sections.iterrows():
            sec = row["section"]
            sec_time = row["section_cumtime_s"]
            sec_share = (sec_time / total_time) * 100.0
            buf.write(f"\nSection: {sec}\n")
            buf.write(f"Total time: {sec_time:.4f}s | Share of total: {sec_share:.2f}%\n")
            buf.write("-" * 80 + "\n")
            sub = df[df["section"] == sec].sort_values("cumtime_s", ascending=False).head(top_k_per_section)
            buf.write(sub[["function", "cumtime_s", "tottime_s", "ncalls_total", "pct_of_section"]]
                      .to_string(index=False))
            buf.write("\n" + "=" * 80 + "\n")

    Path(output_path).write_text(buf.getvalue())
    log.info("[+] Summary written to %s", output_path)


if __name__ == "__main__":
    prof_path = "fit-profile.txt"
    df = parse_profiler_txt(prof_path)
    print_filtered_summary(
        df,
        output_path="summary-profile.txt",
        section_time_thresh=0.2,      # filter out small sections (<0.2s)
        section_share_thresh=5.0,     # or only show sections >=5% of total
        top_k_per_section=3,          # top 3 functions per section
        top_n_sections=7              # top 5 sections overall
    )
    # optional full export
    df.to_csv("profiler_parsed_full2.csv", index=False)

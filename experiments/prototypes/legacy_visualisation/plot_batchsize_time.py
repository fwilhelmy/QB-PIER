import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

from loading import crawl_experiments, load_experiment


# Root directory where your Batch Size experiments are stored.
# Adjust the case to match your filesystem exactly.
ROOT = "logs/VQM-Benchmark/Batch_Size"


def extract_batch_size(path: str) -> int:
    """
    Extract the batch size from a path component like 'BS=128,LR=75e-6'.
    """
    m = re.search(r"BS=(\d+)", path)
    if not m:
        raise ValueError(f"Could not find batch size in path: {path}")
    return int(m.group(1))


def extract_total_time(model_info_path: str) -> float:
    """
    Read `model.info` and extract the 'Total training time (per seed)' in seconds.
    """
    with open(model_info_path, "r") as f:
        lines = f.readlines()

    for line in reversed(lines):
        if "Total training time" in line:
            m = re.search(r"([\d.]+)\s*seconds", line)
            if not m:
                raise ValueError(
                    f"Could not parse total training time from line: {line.strip()}"
                )
            return float(m.group(1))

    raise ValueError(
        f"'Total training time' line not found in {model_info_path}"
    )


def main():
    os.makedirs(ROOT, exist_ok=True)

    # This loads all models (as requested), using the utilities from loading.py.
    # If you only care about timings and not the models themselves, you could
    # skip this call, but it's kept here to "load all the models" explicitly.
    _ = load_experiment(ROOT, load="final")

    # Now walk all experiment directories containing a .model file
    experiment_dirs = crawl_experiments(ROOT)

    # Aggregate times per batch size (in case there are multiple runs per BS)
    times_by_bs = defaultdict(list)

    for exp_dir in experiment_dirs:
        # exp_dir is an absolute path; we only need the last component to get BS
        dirname = os.path.basename(exp_dir)
        bs = extract_batch_size(dirname)

        model_info_path = os.path.join(exp_dir, "model.info")
        if not os.path.exists(model_info_path):
            # You can choose to raise instead if you want this to be strict.
            print(f"[WARN] model.info not found in {exp_dir}, skipping.")
            continue

        total_time = extract_total_time(model_info_path)
        times_by_bs[bs].append(total_time)

    if not times_by_bs:
        raise RuntimeError("No batch-size / training-time data found.")

    # Average over multiple runs for the same batch size
    batch_sizes = sorted(times_by_bs.keys())
    avg_times = [
        sum(times_by_bs[bs]) / len(times_by_bs[bs]) for bs in batch_sizes
    ]

    # Plot
    plt.figure()
    plt.plot(batch_sizes, avg_times, marker="o")
    plt.xlabel("Batch size")
    plt.ylabel("Total training time per seed (seconds)")
    plt.title("Training Time vs Batch Size (VQM-Benchmark)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("batch_size_vs_total_time.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import logging

import pandas as pd

log = logging.getLogger(__name__)

FILE1 = "outputs/file1.csv"
FILE2 = "outputs/file2.csv"
OUT = "outputs/merged.csv"
KEYS = ["seed", "source_model", "target_model", "epsilon"]
SORT_OUTPUT = False

def merge_csv_files(file1: str, file2: str, out: str, keys: list[str], sort_output: bool = False) -> None:
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

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    merge_csv_files(FILE1, FILE2, OUT, KEYS, sort_output=SORT_OUTPUT)

if __name__ == "__main__":
    main()

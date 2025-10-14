#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Union two CSVs, keeping file1's rows on key conflicts.")
    ap.add_argument("file1", help="Origin CSV (authoritative)")
    ap.add_argument("file2", help="CSV to supplement with missing rows")
    ap.add_argument("-o", "--out", default="merged.csv", help="Output CSV path (default: merged.csv)")
    # You can change/extend the default keys if needed.
    ap.add_argument("--keys", nargs="+",
                    default=["seed", "source_model", "target_model", "epsilon"],
                    help="Key columns used to identify unique rows")
    ap.add_argument("--sort", action="store_true", help="Sort output by key columns")
    args = ap.parse_args()

    # Read
    df1 = pd.read_csv(args.file1)
    df2 = pd.read_csv(args.file2)

    # Ensure key columns exist
    missing1 = [k for k in args.keys if k not in df1.columns]
    missing2 = [k for k in args.keys if k not in df2.columns]
    if missing1 or missing2:
        raise ValueError(f"Missing key columns. "
                         f"file1 missing: {missing1}; file2 missing: {missing2}")

    # Put df1 first so its duplicates are kept
    combined = pd.concat([df1, df2], ignore_index=True)

    # Drop duplicates on the key, keeping the first occurrence (which comes from df1)
    combined = combined.drop_duplicates(subset=args.keys, keep="first")

    # Optional: sort for readability
    if args.sort:
        combined = combined.sort_values(args.keys, kind="stable")

    combined.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(combined)} rows "
          f"(file1 rows preserved on key conflicts).")

if __name__ == "__main__":
    main()

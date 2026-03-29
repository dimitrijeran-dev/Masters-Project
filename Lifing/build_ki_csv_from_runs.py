#!/usr/bin/env python3
import argparse
import json
import csv
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--a-values", nargs="+", type=float, required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    if len(args.run_dirs) != len(args.a_values):
        raise ValueError(
            "Length mismatch: --run-dirs and --a-values must contain the same "
            f"number of entries (got {len(args.run_dirs)} run dirs and "
            f"{len(args.a_values)} a-values)."
        )

    validation_paths = [Path(run_dir) / "validation_summary.json" for run_dir in args.run_dirs]
    missing_files = [str(path) for path in validation_paths if not path.is_file()]
    if missing_files:
        missing_list = "\n  - ".join(missing_files)
        raise FileNotFoundError(
            "Missing required validation summary files:\n"
            f"  - {missing_list}"
        )

    rows = []

    for path, a in zip(validation_paths, args.a_values):
        with open(path) as f:
            data = json.load(f)

        rows.append({
            "a": a,
            "KI": data["KI_ref"]
        })

    rows.sort(key=lambda r: r["a"])

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["a", "KI"])
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote", args.out_csv)

if __name__ == "__main__":
    main()

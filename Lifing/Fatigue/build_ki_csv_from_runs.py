
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

    rows = []

    for run_dir, a in zip(args.run_dirs, args.a_values):
        path = Path(run_dir) / "validation_summary.json"
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

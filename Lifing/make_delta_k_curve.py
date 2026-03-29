#!/usr/bin/env python3
import argparse
import pandas as pd
from fatigue_lifing_utils import load_ki_csv, compute_delta_k_from_R, compute_delta_k
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.run_manifest import load_run_manifest, write_run_manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ki-max-csv", required=True)
    parser.add_argument("--ki-min-csv")
    parser.add_argument("--R", type=float)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_path = outdir / "run_manifest.json"
    if manifest_path.exists():
        manifest = load_run_manifest(outdir)
        manifest_hash = manifest.get("manifest_hash_sha256")
    else:
        _, manifest_hash = write_run_manifest(
            outdir,
            {
                "workflow": "Lifing.make_delta_k_curve",
                "lifing": {
                    "ki_max_csv": args.ki_max_csv,
                    "ki_min_csv": args.ki_min_csv,
                    "R": args.R,
                },
                "rng": {"seed_derivation_rule": "deterministic_no_rng"},
            },
        )

    df_max = load_ki_csv(args.ki_max_csv)

    if args.ki_min_csv:
        df_min = load_ki_csv(args.ki_min_csv)
        delta_k = compute_delta_k(df_max["KI"], df_min["KI"])
    else:
        if args.R is None:
            raise ValueError("Provide either --ki-min-csv or --R")
        delta_k = compute_delta_k_from_R(df_max["KI"], args.R)

    out = pd.DataFrame({
        "a": df_max["a"],
        "DeltaK": delta_k
    })

    out.to_csv(outdir / "delta_k_curve.csv", index=False)
    (outdir / "delta_k_summary.json").write_text(
        json.dumps(
            {
                "n_points": int(len(out)),
                "manifest_hash_sha256": manifest_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Wrote delta_k_curve.csv")

if __name__ == "__main__":
    main()

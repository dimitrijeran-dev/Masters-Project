#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from fatigue_lifing_utils import (
    load_ki_csv,
    compute_delta_k_from_R,
    compute_delta_k,
)

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

        if len(df_max) != len(df_min):
            raise ValueError("KI max/min CSVs must have same number of rows")
        if not (df_max["a"].values == df_min["a"].values).all():
            raise ValueError("KI max/min CSVs must share the same a-grid")

        kmax = df_max["KI"].values
        kmin = df_min["KI"].values
        delta_k = compute_delta_k(kmax, kmin)

        # effective R pointwise where possible
        with pd.option_context("mode.use_inf_as_na", True):
            R_eff = pd.Series(kmin / kmax).replace([pd.NA], pd.NA)

        out = pd.DataFrame({
            "a": df_max["a"].values,
            "Kmax": kmax,
            "Kmin": kmin,
            "DeltaK": delta_k,
            "R": R_eff,
        })

    else:
        if args.R is None:
            raise ValueError("Provide either --ki-min-csv or --R")

        kmax = df_max["KI"].values
        kmin = args.R * kmax
        delta_k = compute_delta_k_from_R(kmax, args.R)

        out = pd.DataFrame({
            "a": df_max["a"].values,
            "Kmax": kmax,
            "Kmin": kmin,
            "DeltaK": delta_k,
            "R": args.R,
        })

    out.to_csv(outdir / "delta_k_curve.csv", index=False)

    (outdir / "delta_k_summary.json").write_text(
        json.dumps(
            {
                "n_points": int(len(out)),
                "has_kmax": True,
                "has_kmin": True,
                "manifest_hash_sha256": manifest_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Wrote delta_k_curve.csv with columns: a, Kmax, Kmin, DeltaK, R")


if __name__ == "__main__":
    main()
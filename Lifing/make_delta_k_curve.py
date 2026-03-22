#!/usr/bin/env python3
import argparse
import pandas as pd
from fatigue_lifing_utils import load_ki_csv, compute_delta_k_from_R, compute_delta_k

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ki-max-csv", required=True)
    parser.add_argument("--ki-min-csv")
    parser.add_argument("--R", type=float)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

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

    import os
    os.makedirs(args.outdir, exist_ok=True)
    out.to_csv(f"{args.outdir}/delta_k_curve.csv", index=False)
    print("Wrote delta_k_curve.csv")

if __name__ == "__main__":
    main()

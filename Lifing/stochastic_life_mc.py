#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fatigue_lifing_utils import integrate_crack_growth
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.run_manifest import load_run_manifest, write_run_manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta-k-csv", required=True)
    parser.add_argument("--C-mean", type=float, required=True)
    parser.add_argument("--C-cov", type=float, required=True)
    parser.add_argument("--m-mean", type=float, required=True)
    parser.add_argument("--m-std", type=float, required=True)
    parser.add_argument("--sigma-scale-mean", type=float, required=True)
    parser.add_argument("--sigma-scale-cov", type=float, required=True)
    parser.add_argument("--nsamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
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
                "workflow": "Lifing.stochastic_life_mc",
                "lifing": {
                    "delta_k_csv": args.delta_k_csv,
                    "C_mean": args.C_mean,
                    "C_cov": args.C_cov,
                    "m_mean": args.m_mean,
                    "m_std": args.m_std,
                    "sigma_scale_mean": args.sigma_scale_mean,
                    "sigma_scale_cov": args.sigma_scale_cov,
                    "nsamples": args.nsamples,
                },
                "rng": {
                    "base_seed": args.seed,
                    "seed_derivation_rule": "sample_seed_i = base_seed + i",
                    "per_realization": [{"realization_id": i, "seed": args.seed + i} for i in range(args.nsamples)],
                },
            },
        )

    df = pd.read_csv(args.delta_k_csv)
    a = df["a"].values
    delta_k = df["DeltaK"].values / 1e6  # convert Pa√m → MPa√m

    rng = np.random.default_rng(args.seed)
    C_samples = rng.normal(args.C_mean, args.C_mean*args.C_cov, args.nsamples)
    m_samples = rng.normal(args.m_mean, args.m_std, args.nsamples)

    life = []

    for C, m in zip(C_samples, m_samples):
        N, _ = integrate_crack_growth(a, delta_k, C, m)
        life.append(N[-1])

    life = np.array(life)

    import os
    os.makedirs(args.outdir, exist_ok=True)

    plt.figure()
    plt.hist(np.log10(life), bins=40)
    plt.xlabel("log10(Cycles to failure)")
    plt.ylabel("Frequency")
    plt.title("Monte Carlo fatigue life")
    plt.savefig(f"{args.outdir}/life_histogram.png")
    (outdir / "life_summary.json").write_text(
        json.dumps(
            {
                "nsamples": args.nsamples,
                "seed": args.seed,
                "life_log10_mean": float(np.mean(np.log10(life))),
                "life_log10_std": float(np.std(np.log10(life))),
                "manifest_hash_sha256": manifest_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Monte Carlo simulation complete")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fatigue_lifing_utils import integrate_crack_growth


def _safe_cov(std: float, mean: float) -> float:
    if abs(mean) < 1e-30:
        return float("nan")
    return float(std / mean)


def _stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "mean": mean,
        "median": float(np.median(arr)),
        "std": std,
        "cov": _safe_cov(std, mean),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "skew": float(pd.Series(arr).skew()),
    }


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
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--run-name", default="stochastic_lifing")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.delta_k_csv)
    a = df["a"].values
    delta_k = df["DeltaK"].values / 1e6  # convert Pa√m → MPa√m

    C_samples = rng.normal(args.C_mean, args.C_mean * args.C_cov, args.nsamples)
    m_samples = rng.normal(args.m_mean, args.m_std, args.nsamples)
    sigma_scale_samples = rng.normal(
        args.sigma_scale_mean,
        args.sigma_scale_mean * args.sigma_scale_cov,
        args.nsamples,
    )

    life = []

    for C, m, sigma_scale in zip(C_samples, m_samples, sigma_scale_samples):
        N, _ = integrate_crack_growth(a, delta_k * sigma_scale, C, m)
        life.append(N[-1])

    life = np.array(life)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(np.log10(life), bins=40)
    plt.xlabel("log10(Cycles to failure)")
    plt.ylabel("Frequency")
    plt.title("Monte Carlo fatigue life")
    plt.savefig(outdir / "life_histogram.png")

    samples_df = pd.DataFrame(
        {
            "sample_id": np.arange(args.nsamples, dtype=int),
            "C": C_samples,
            "m": m_samples,
            "sigma_scale": sigma_scale_samples,
            "life_cycles": life,
        }
    )
    samples_df.to_csv(outdir / "life_samples.csv", index=False)

    corr = samples_df[["C", "m", "sigma_scale", "life_cycles"]].corr(method="pearson")

    summary = {
        "metadata": {
            "run_name": args.run_name,
            "seed": int(args.seed),
            "nsamples": int(args.nsamples),
            "units": {
                "a": "m",
                "DeltaK_input": "Pa*sqrt(m)",
                "DeltaK_used_in_life": "MPa*sqrt(m)",
                "C": "(cycle^-1) * (MPa*sqrt(m))^-m",
                "m": "unitless",
                "sigma_scale": "unitless",
                "life_cycles": "cycles",
            },
        },
        "life_distribution": _stats(life),
        "sampled_parameter_correlations": corr.to_dict(),
    }

    (outdir / "life_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Monte Carlo simulation complete")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fatigue_lifing_utils import (
    integrate_crack_growth,
    find_critical_crack_length,
    truncate_at_acrit,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.run_manifest import load_run_manifest, write_run_manifest


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
    parser.add_argument("--run-name", default="stochastic_lifing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--KIC",
        type=float,
        default=None,
        help=(
            "Optional fracture toughness in MPa*sqrt(m). "
            "If provided, each Monte Carlo sample computes its own "
            "a_crit from sigma_scale*Kmax(a) = KIC and truncates life at failure."
        ),
    )
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
                    "KIC": args.KIC,
                },
                "rng": {
                    "base_seed": args.seed,
                    "seed_derivation_rule": "sample_seed_i = base_seed + i",
                    "per_realization": [
                        {"realization_id": i, "seed": args.seed + i}
                        for i in range(args.nsamples)
                    ],
                },
            },
        )

    df = pd.read_csv(args.delta_k_csv)

    required_cols = {"a", "DeltaK"}
    if not required_cols.issubset(df.columns):
        raise ValueError("delta_k CSV must contain at least columns: a, DeltaK")

    a = df["a"].to_numpy(dtype=float)
    delta_k = df["DeltaK"].to_numpy(dtype=float) / 1e6  # Pa*sqrt(m) -> MPa*sqrt(m)

    # Needed only if KIC-based sample-specific failure is requested
    if args.KIC is not None:
        if "Kmax" not in df.columns:
            raise ValueError(
                "KIC was provided, but delta_k CSV does not contain Kmax. "
                "Rebuild delta_k_curve.csv so it includes columns: a, Kmax, Kmin, DeltaK, R."
            )
        kmax_baseline = df["Kmax"].to_numpy(dtype=float) / 1e6  # Pa*sqrt(m) -> MPa*sqrt(m)
    else:
        kmax_baseline = None

    rng = np.random.default_rng(args.seed)

    C_samples = rng.normal(args.C_mean, args.C_mean * args.C_cov, args.nsamples)
    m_samples = rng.normal(args.m_mean, args.m_std, args.nsamples)
    sigma_scale_samples = rng.normal(
        args.sigma_scale_mean,
        args.sigma_scale_mean * args.sigma_scale_cov,
        args.nsamples,
    )

    life = []
    acrit_list = []
    failed_within_grid = []

    for i, (C, m, sigma_scale) in enumerate(zip(C_samples, m_samples, sigma_scale_samples)):
        delta_k_sample = delta_k * sigma_scale

        # Base integration on the full grid first
        N_full, dadn_full = integrate_crack_growth(a, delta_k_sample, C, m)

        if args.KIC is None:
            life_i = float(N_full[-1])
            acrit_i = float("nan")
            failed_i = False
        else:
            kmax_sample = kmax_baseline * sigma_scale
            acrit_i, idx_fail = find_critical_crack_length(a, kmax_sample, args.KIC)

            if acrit_i is None:
                # fracture criterion not reached over supplied a-grid
                life_i = float(N_full[-1])
                acrit_i = float("nan")
                failed_i = False
            else:
                Ncrit_i = float(np.interp(acrit_i, a, N_full))
                life_i = Ncrit_i
                failed_i = True

        life.append(life_i)
        acrit_list.append(acrit_i)
        failed_within_grid.append(failed_i)

    life = np.asarray(life, dtype=float)
    acrit_arr = np.asarray(acrit_list, dtype=float)
    failed_within_grid = np.asarray(failed_within_grid, dtype=bool)

    # Histogram
    positive_life = life[np.isfinite(life) & (life > 0)]
    if positive_life.size == 0:
        raise ValueError("No positive fatigue lives were computed.")

    plt.figure()
    plt.hist(np.log10(positive_life), bins=40)
    plt.ylabel("Frequency")
    if args.KIC is None:
        plt.title("Monte Carlo Fatigue Life")
        plt.xlabel("log10(Cycles Over Full Crack Length Grid)")
    else:
        plt.title("Monte Carlo Fatigue Life (with KIC cutoff)")
        plt.xlabel("log10(Cycles to failure)")
    plt.tight_layout()
    plt.savefig(outdir / "life_histogram.png", dpi=300)
    plt.close()

    samples_df = pd.DataFrame(
        {
            "sample_id": np.arange(args.nsamples, dtype=int),
            "C": C_samples,
            "m": m_samples,
            "sigma_scale": sigma_scale_samples,
            "life_cycles": life,
            "a_crit_m": acrit_arr,
            "failed_within_grid": failed_within_grid,
        }
    )
    samples_df.to_csv(outdir / "life_samples.csv", index=False)

    corr_cols = ["C", "m", "sigma_scale", "life_cycles"]
    if np.any(np.isfinite(acrit_arr)):
        corr_cols.append("a_crit_m")
    corr = samples_df[corr_cols].corr(method="pearson")

    summary = {
        "metadata": {
            "run_name": args.run_name,
            "seed": int(args.seed),
            "nsamples": int(args.nsamples),
            "KIC_MPa_sqrt_m": args.KIC,
            "manifest_hash_sha256": manifest_hash,
            "units": {
                "a": "m",
                "a_crit_m": "m",
                "DeltaK_input": "Pa*sqrt(m)",
                "DeltaK_used_in_life": "MPa*sqrt(m)",
                "Kmax_used_for_failure": "MPa*sqrt(m)",
                "KIC": "MPa*sqrt(m)",
                "C": "(cycle^-1) * (MPa*sqrt(m))^-m",
                "m": "unitless",
                "sigma_scale": "unitless",
                "life_cycles": "cycles",
            },
        },
        "life_distribution": _stats(life),
        "n_failed_within_grid": int(np.sum(failed_within_grid)),
        "fraction_failed_within_grid": float(np.mean(failed_within_grid)),
        "sampled_parameter_correlations": corr.to_dict(),
    }

    finite_acrit = acrit_arr[np.isfinite(acrit_arr)]
    if finite_acrit.size > 0:
        summary["acrit_distribution"] = _stats(finite_acrit)

    (outdir / "life_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Monte Carlo simulation complete")
    if args.KIC is None:
        print("No KIC cutoff applied; life integrated over the full supplied crack-length grid.")
    else:
        print(f"KIC cutoff applied: {args.KIC:.6g} MPa*sqrt(m)")
        print(f"Samples failing within supplied grid: {np.sum(failed_within_grid)}/{args.nsamples}")


if __name__ == "__main__":
    main()
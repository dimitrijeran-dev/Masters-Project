#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fatigue_lifing_utils import integrate_crack_growth

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
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.delta_k_csv)
    a = df["a"].values
    delta_k = df["DeltaK"].values / 1e6  # convert Pa√m → MPa√m

    C_samples = np.random.normal(args.C_mean, args.C_mean*args.C_cov, args.nsamples)
    m_samples = np.random.normal(args.m_mean, args.m_std, args.nsamples)

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

    print("Monte Carlo simulation complete")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from fatigue_lifing_utils import paris_law, integrate_crack_growth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta-k-csv", required=True)
    parser.add_argument("--C", type=float, required=True)
    parser.add_argument("--m", type=float, required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.delta_k_csv)
    a = df["a"].values
    delta_k = df["DeltaK"].values / 1e6  # convert Pa√m → MPa√m

    dadn = paris_law(delta_k, args.C, args.m)
    N, dadn = integrate_crack_growth(a, delta_k, args.C, args.m)

    import os
    os.makedirs(args.outdir, exist_ok=True)

    plt.figure()
    plt.loglog(delta_k, dadn, marker="o")
    plt.xlabel("Delta K")
    plt.ylabel("da/dN")
    plt.title("Paris law curve")
    plt.savefig(f"{args.outdir}/dadn_vs_delta_k.png")

    plt.figure()
    plt.plot(a, dadn)
    plt.xlabel("Crack length a")
    plt.ylabel("da/dN")
    plt.savefig(f"{args.outdir}/dadn_vs_a.png")

    plt.figure()
    plt.plot(N, a)
    plt.xlabel("Cycles N")
    plt.ylabel("Crack length a")
    plt.savefig(f"{args.outdir}/a_vs_N.png")

    print("Generated Paris law plots")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fatigue_lifing_utils import paris_law, integrate_crack_growth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta-k-csv", required=True)
    parser.add_argument("--C", type=float, required=True)
    parser.add_argument("--m", type=float, required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--delta-k-th",
        type=float,
        default=None,
        help="Threshold Delta K in MPa*sqrt(m). Example: 3.715"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.delta_k_csv)
    a = df["a"].values
    delta_k = df["DeltaK"].values / 1e6  # convert Pa*sqrt(m) -> MPa*sqrt(m)

    # Standard Paris-law evaluation
    dadn = paris_law(delta_k, args.C, args.m)
    N, dadn_integrated = integrate_crack_growth(a, delta_k, args.C, args.m)

    # For plotting only: hide Paris-law values below threshold if provided
    dadn_plot = dadn.copy()
    if args.delta_k_th is not None:
        dadn_plot = np.where(delta_k >= args.delta_k_th, dadn_plot, np.nan)

    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------
    # da/dN vs Delta K
    # ------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.loglog(delta_k, dadn_plot, marker="o", label="Paris law")

    if args.delta_k_th is not None:
        ax.axvline(
            args.delta_k_th,
            linestyle="--",
            linewidth=1.8,
            label=fr"$\Delta K_{{th}} = {args.delta_k_th:.3f}$ MPa$\sqrt{{m}}$"
        )

        # Place annotation using the current y-limits after plotting
        finite_vals = dadn_plot[np.isfinite(dadn_plot) & (dadn_plot > 0)]
        if finite_vals.size > 0:
            y_annot = finite_vals.min() * 1.5
        else:
            y_annot = 1e-12

        ax.text(
            args.delta_k_th * 1.03,
            y_annot,
            fr"$\log_{{10}}(\Delta K_{{th}}) = {np.log10(args.delta_k_th):.3f}$",
            rotation=90,
            va="bottom",
            ha="left"
        )

    ax.set_xlabel(r"$\Delta K$ (MPa$\sqrt{m}$)")
    ax.set_ylabel(r"$da/dN$")
    ax.set_title("Paris law curve")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/dadn_vs_delta_k.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------
    # da/dN vs a
    # ------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(a, dadn_integrated, marker="o")
    ax.set_xlabel("Crack length a")
    ax.set_ylabel("da/dN")
    ax.set_title("Crack growth rate vs crack length")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/dadn_vs_a.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------
    # a vs N
    # ------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(N, a)
    ax.set_xlabel("Cycles N")
    ax.set_ylabel("Crack length a")
    ax.set_title("Crack length vs cycles")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/a_vs_N.png", dpi=300)
    plt.close(fig)

    print("Generated Paris law plots")
    if args.delta_k_th is not None:
        print(f"Included threshold line at DeltaK_th = {args.delta_k_th:.3f} MPa*sqrt(m)")
        print(f"log10(DeltaK_th) = {np.log10(args.delta_k_th):.3f}")

if __name__ == "__main__":
    main()
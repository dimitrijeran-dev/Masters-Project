#!/usr/bin/env python3
import os
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fatigue_lifing_utils import (
    paris_law,
    integrate_crack_growth,
    find_critical_crack_length,
    truncate_at_acrit,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.run_manifest import load_run_manifest, write_run_manifest


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
        help="Threshold Delta K in MPa*sqrt(m). Example: 3.715",
    )
    parser.add_argument(
        "--KIC",
        type=float,
        default=None,
        help="Fracture toughness in MPa*sqrt(m). If provided, compute a_crit from Kmax(a)=KIC and truncate curves at failure.",
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
                "workflow": "Lifing.make_dadn_curve",
                "lifing": {
                    "delta_k_csv": args.delta_k_csv,
                    "C": args.C,
                    "m": args.m,
                    "delta_k_th": args.delta_k_th,
                    "KIC": args.KIC,
                },
                "rng": {"seed_derivation_rule": "deterministic_no_rng"},
            },
        )

    df = pd.read_csv(args.delta_k_csv)

    required_cols = {"a", "DeltaK"}
    if not required_cols.issubset(df.columns):
        raise ValueError("delta-k CSV must contain at least: a, DeltaK")

    a_mm = df["a"].values
    a = a_mm / 1e3  # convert mm -> m
    
    delta_k = df["DeltaK"].values / 1e6  # Pa*sqrt(m) -> MPa*sqrt(m)

    if "Kmax" in df.columns:
        kmax = df["Kmax"].values / 1e6  # Pa*sqrt(m) -> MPa*sqrt(m)
    else:
        kmax = None

    dadn = paris_law(delta_k, args.C, args.m)
    N, dadn_integrated = integrate_crack_growth(a, delta_k, args.C, args.m)

    # Plot-only threshold masking
    dadn_plot = dadn.copy()
    if args.delta_k_th is not None:
        dadn_plot = np.where(delta_k >= args.delta_k_th, dadn_plot, np.nan)

    # -----------------------------
    # Critical crack length from KIC
    # -----------------------------
    acrit = None
    idx_fail = None
    Ncrit = None

    if args.KIC is not None:
        if kmax is None:
            raise ValueError(
                "KIC was provided, but delta_k CSV does not contain Kmax. "
                "Rebuild delta_k_curve.csv with Kmax included."
            )

        acrit, idx_fail = find_critical_crack_length(a, kmax, args.KIC)

        if acrit is not None:
            Ncrit = float(np.interp(acrit, a, N))

    # Truncated curves for plotting/output
    a_plot, dadn_vs_a_plot = truncate_at_acrit(a_mm, dadn_integrated, acrit)
    N_plot, a_vs_N_plot = truncate_at_acrit(N, a_mm, Ncrit) if Ncrit is not None else (N, a)

    # -----------------------------
    # da/dN vs DeltaK
    # -----------------------------
    fig, ax = plt.subplots()
    ax.loglog(delta_k, dadn_plot, marker="o", label="Paris law")

    if args.delta_k_th is not None:
        ax.axvline(
            args.delta_k_th,
            linestyle="--",
            linewidth=1.8,
            label=fr"$\Delta K_{{th}} = {args.delta_k_th:.3f}$ MPa$\sqrt{{m}}$",
        )

        finite_vals = dadn_plot[np.isfinite(dadn_plot) & (dadn_plot > 0)]
        y_annot = finite_vals.min() * 1.5 if finite_vals.size > 0 else 1e-12
        ax.text(
            args.delta_k_th * 1.03,
            y_annot,
            fr"$\log_{{10}}(\Delta K_{{th}}) = {np.log10(args.delta_k_th):.3f}$",
            rotation=90,
            va="bottom",
            ha="left",
        )

    ax.set_xlabel(r"$\Delta K$ (MPa$\sqrt{m}$)")
    ax.set_ylabel(r"$da/dN$ (mm/cycle)")
    ax.set_title("Paris law curve")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/dadn_vs_delta_k.png", dpi=300)
    plt.close(fig)

    # -----------------------------
    # da/dN vs a
    # -----------------------------
    fig, ax = plt.subplots()
    ax.plot(a_plot, dadn_vs_a_plot, marker="o")

    if acrit is not None:
        ax.axvline(acrit, linestyle="--", linewidth=1.8, label=fr"$a_{{crit}}={acrit:.6g}$ m")
        ax.legend()

    ax.set_xlabel("Crack length a (mm)")
    ax.set_ylabel("da/dN (mm/cycle)")
    ax.set_title("Crack growth rate vs crack length")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/dadn_vs_a.png", dpi=300)
    plt.close(fig)

    # -----------------------------
    # a vs N
    # -----------------------------
    fig, ax = plt.subplots()
    ax.plot(N_plot, a_vs_N_plot)

    if Ncrit is not None:
        ax.axvline(Ncrit, linestyle="--", linewidth=1.8, label=fr"$N_f={Ncrit:.6g}$")
        ax.legend()

    ax.set_xlabel("Cycles N")
    ax.set_ylabel("Crack length a (mm)")
    ax.set_title("Crack Length vs Cycles")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/a_vs_N.png", dpi=300)
    plt.close(fig)

    # Save truncated data too
    pd.DataFrame({
        "a": a_plot,
        "dadn": dadn_vs_a_plot,
    }).to_csv(outdir / "dadn_vs_a_truncated.csv", index=False)

    pd.DataFrame({
        "N": N_plot,
        "a": a_vs_N_plot,
    }).to_csv(outdir / "a_vs_N_truncated.csv", index=False)

    summary = {
        "n_points": int(len(a)),
        "delta_k_th": args.delta_k_th,
        "KIC": args.KIC,
        "a_crit_m": acrit,
        "N_crit_cycles": Ncrit,
        "manifest_hash_sha256": manifest_hash,
    }

    (outdir / "dadn_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Generated Paris law plots")
    if args.delta_k_th is not None:
        print(f"Included threshold line at DeltaK_th = {args.delta_k_th:.3f} MPa*sqrt(m)")
        print(f"log10(DeltaK_th) = {np.log10(args.delta_k_th):.3f}")

    if args.KIC is not None:
        if acrit is None:
            print("Kmax(a) never reached KIC over supplied crack-length range.")
        else:
            print(f"Critical crack length a_crit = {acrit:.6e} m")
            print(f"Failure cycles N_crit = {Ncrit:.6e}")


if __name__ == "__main__":
    main()
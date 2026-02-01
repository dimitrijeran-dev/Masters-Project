#!/usr/bin/env python3
"""
Compute and plot geometry factor F versus a_eff/b using measured K_I(a):

    a_eff = a + R
    F(a_eff/b) = K_I(a) / [ sigma_inf * sqrt(pi * a_eff) ]

Inputs:
  - KI_summary.csv from extract_KI_vs_a.py
  - b (mm), R (mm), and sigma_inf (MPa)

Outputs:
  - F_vs_ab.csv
  - F_vs_ab.png
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ki-summary", type=Path, required=True, help="KI_summary.csv")
    ap.add_argument("--b-mm", type=float, default=100.0, help="b in mm (default 100)")
    ap.add_argument("--R-mm", type=float, default=15.0, help="Hole radius R in mm (default 15 for Ø30)")
    ap.add_argument("--sigma-mpa", type=float, default=25.0,
                    help="Remote applied stress sigma_inf in MPa (default 25 MPa for 10kN, t=2mm, width=200mm)")
    ap.add_argument("--out-csv", type=Path, default=Path("F_vs_ab.csv"))
    ap.add_argument("--out-png", type=Path, default=Path("F_vs_ab.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.ki_summary)
    if "a_mm" not in df.columns or "KI_Pa_sqrt_m" not in df.columns:
        raise SystemExit("KI_summary.csv must contain columns: a_mm, KI_Pa_sqrt_m")

    # ---- Effective crack length ----
    a_mm = df["a_mm"].astype(float).to_numpy()
    a_eff_mm = a_mm + float(args.R_mm)

    # Units
    a_eff_m = a_eff_mm * 1e-3
    b_m = float(args.b_mm) * 1e-3
    sigma_pa = float(args.sigma_mpa) * 1e6

    KI = df["KI_Pa_sqrt_m"].astype(float).to_numpy()  # Pa*sqrt(m)

    # ---- Geometry factor ----
    ab_eff = a_eff_mm / float(args.b_mm)
    F = KI / (sigma_pa * np.sqrt(np.pi * a_eff_m))

    out = pd.DataFrame({
        "a_mm": a_mm,
        "R_mm": float(args.R_mm),
        "a_eff_mm": a_eff_mm,
        "a_eff_over_b": ab_eff,
        "KI_MPa_sqrt_m": KI / 1e6,
        "sigma_inf_MPa": float(args.sigma_mpa),
        "F": F,
    }).sort_values("a_mm").reset_index(drop=True)

    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv.resolve()}")
    print(out[["a_mm", "a_eff_mm", "a_eff_over_b", "F"]])

    # ---- Smooth curve for visualization ----
    x = out["a_eff_over_b"].to_numpy()
    y = out["F"].to_numpy()

    deg = min(3, len(x) - 1)
    coeff = np.polyfit(x, y, deg=deg)
    poly = np.poly1d(coeff)

    xx = np.linspace(x.min(), x.max(), 200)
    yy = poly(xx)

    # Plot
    plt.figure()
    plt.plot(xx, yy, label=f"Poly fit (deg={deg})")
    plt.scatter(x, y, s=60, label="DCM-derived points")

    for ai, xi, yi in zip(out["a_mm"], x, y):
        plt.annotate(f"{ai:.0f} mm", (xi, yi),
                     textcoords="offset points", xytext=(6, 6))

    plt.xlabel("a_eff / b   with  a_eff = a + R")
    plt.ylabel("F   from  K = σ√(π a_eff) F")
    plt.title("Geometry factor from DCM-extracted K_I (hole + crack)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"Saved plot: {args.out_png.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

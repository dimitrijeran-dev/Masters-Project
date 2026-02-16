#!/usr/bin/env python3
"""
Convert pyFEM nodal CSV (node_id,x,y,ux,uy) into DCM simple CSV:
    r_m, uy_upper_m, uy_lower_m

Assumes:
- crack line is y=0
- upper face nodes have y > 0, lower face nodes have y < 0
- pairing is done by nearest x (within a tolerance)
- r is computed as (x_tip - x_pair)

Also optionally plots KI(r) using the E'/8 COD relation (same as your DCM.py).
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True, help="pyFEM_nodes_for_dcm.csv")
    ap.add_argument("--out-csv", default="dcm_input.csv", help="Output CSV with r_m,uy_upper_m,uy_lower_m")
    ap.add_argument("--x-tip", type=float, required=True, help="Crack tip x-coordinate (m)")
    ap.add_argument("--y-tip", type=float, default=0.0, help="Crack line y-coordinate (m) (default 0)")
    ap.add_argument("--y-band", type=float, default=1e-3, help="Half band about crack line to keep nodes (m)")
    ap.add_argument("--x-match-tol", type=float, default=2e-4, help="Pairing tolerance in x (m)")
    ap.add_argument("--plot", action="store_true", help="Plot KI vs r (using E'/8 COD relation)")
    ap.add_argument("--E", type=float, default=7.31e10, help="Young's modulus (Pa)")
    ap.add_argument("--nu", type=float, default=0.33, help="Poisson ratio")
    ap.add_argument("--plane-strain", action="store_true", help="Use plane strain E' = E/(1-nu^2)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    # Keep only nodes near the crack line (your exporter already does this, but keep robust)
    df = df[np.abs(df["y"] - args.y_tip) <= args.y_band].copy()

    upper = df[df["y"] > args.y_tip].copy()
    lower = df[df["y"] < args.y_tip].copy()

    if upper.empty or lower.empty:
        raise SystemExit("Not enough upper/lower nodes. Increase y-band or ensure slit has two faces.")

    upper.sort_values("x", inplace=True)
    lower.sort_values("x", inplace=True)

    lower_x = lower["x"].to_numpy()
    lower_uy = lower["uy"].to_numpy()

    pairs = []
    for _, urow in upper.iterrows():
        xu = float(urow["x"])
        j = int(np.searchsorted(lower_x, xu))
        cand = []
        if 0 <= j < len(lower_x): cand.append(j)
        if 0 <= j - 1 < len(lower_x): cand.append(j - 1)

        best = None
        best_dx = None
        for c in cand:
            dx = abs(float(lower_x[c]) - xu)
            if best is None or dx < best_dx:
                best = c
                best_dx = dx

        if best is not None and best_dx is not None and best_dx <= args.x_match_tol:
            xpair = 0.5 * (xu + float(lower_x[best]))
            uy_u = float(urow["uy"])
            uy_l = float(lower_uy[best])
            r = float(args.x_tip - xpair)
            if r > 0:
                pairs.append((r, uy_u, uy_l))

    if not pairs:
        raise SystemExit("No pairs formed. Increase x-match-tol and/or y-band.")

    out = pd.DataFrame(pairs, columns=["r_m", "uy_upper_m", "uy_lower_m"])
    out.sort_values("r_m", inplace=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}  (rows={len(out)})")

    if args.plot:
        # E' for COD relation
        Eprime = args.E / (1.0 - args.nu**2) if args.plane_strain else args.E
        cod = out["uy_upper_m"].to_numpy() - out["uy_lower_m"].to_numpy()
        r = out["r_m"].to_numpy()
        KI = (Eprime / 8.0) * cod * np.sqrt(2.0 * np.pi / r)  # Pa*sqrt(m)

        plt.figure()
        plt.plot(r, KI / 1e6, marker="o")
        plt.gca().invert_xaxis()  # optional: closer to tip is smaller r
        plt.xlabel("r (m)")
        plt.ylabel("K_I (MPa*sqrt(m))")
        plt.grid(True)
        plt.tight_layout()
        png = "pyFEM_KI_vs_r.png"
        plt.savefig(png, dpi=200)
        print(f"Saved plot: {png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

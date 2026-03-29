#!/usr/bin/env python3
"""
DCM utilities for Mode I stress intensity factor estimation.

This version supports three workflows:

1) Direct Mode I calculation from crack-face displacements at a known radial
   location (CLI flags).
2) Batch Mode I calculation from a simple CSV with columns:
      r_m, uy_upper_m, uy_lower_m
3) Automatic pairing of upper/lower crack-face nodes exported from Siemens NX
   (or similar) followed by a radial sweep to estimate the K_I plateau.

Features:
- Interactive prompts: you can run `./src/dcm.py` and it will ask for missing inputs.
- Robust NX column mapping (handles "Node ID", "X Coord", "Y Coord", "Y").
- Units handling for NX exports (mm or m). Recommended: export in mm and set --nx-units mm
  (default), and the script converts to meters internally so K is in MPa*sqrt(m).
- Optional automatic y_tip / x_tip detection (--auto-tip).
- NEW: Automatic r-window selection (r_min / r_max) from the paired K_I(r) data:
    * If user does not provide r-min and/or r-max, the code selects a contiguous
      “flattest” window of K_I(r) via a sliding linear fit (min slope + low scatter).
    * This removes the need for default r-min/r-max values and adapts per crack length.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


# -----------------------------
# Interactive prompting helpers
# -----------------------------
def _prompt_str(name: str, default: Optional[str] = None) -> str:
    if default is not None:
        txt = input(f"{name} [{default}]: ").strip()
        return txt if txt else default
    return input(f"{name}: ").strip()


def _prompt_float(name: str, default: Optional[float] = None) -> float:
    while True:
        try:
            if default is not None:
                txt = input(f"{name} [{default}]: ").strip()
                return float(txt) if txt else float(default)
            return float(input(f"{name}: ").strip())
        except ValueError:
            print("  -> Please enter a numeric value.")


def _prompt_bool(name: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    txt = input(f"{name} [{d}]: ").strip().lower()
    if not txt:
        return default
    return txt in ("y", "yes", "true", "1")


# -----------------------------
# Data containers
# -----------------------------
@dataclass(frozen=True)
class CrackFaceDisplacement:
    """Crack-face displacement pair at a radial distance from the crack tip."""
    r: float
    uy_upper: float
    uy_lower: float

    def delta_uy(self) -> float:
        return self.uy_upper - self.uy_lower


@dataclass(frozen=True)
class Material:
    """Material properties needed for DCM calculations."""
    elastic_modulus: float  # Young's modulus (Pa)
    poisson_ratio: float    # Poisson's ratio
    plane_strain: bool = False

    def shear_modulus(self) -> float:
        return self.elastic_modulus / (2.0 * (1.0 + self.poisson_ratio))

    def kappa(self) -> float:
        if self.plane_strain:
            return 3.0 - 4.0 * self.poisson_ratio
        return (3.0 - self.poisson_ratio) / (1.0 + self.poisson_ratio)

    def effective_modulus(self) -> float:
        # E' (E for plane stress, E/(1-ν²) for plane strain)
        if self.plane_strain:
            return self.elastic_modulus / (1.0 - self.poisson_ratio**2)
        return self.elastic_modulus


@dataclass(frozen=True)
class ModeIResult:
    sif: float
    displacement: CrackFaceDisplacement
    material: Material


# -----------------------------
# Core calculations
# -----------------------------
def compute_mode_i_sif(displacement: CrackFaceDisplacement, material: Material) -> ModeIResult:
    """
    Compute Mode I stress intensity factor using DCM relation:

      K_I = [2 μ / (κ + 1)] * Δu_y * sqrt(2π / r)
    """
    if displacement.r <= 0:
        raise ValueError("Radial distance r must be positive for DCM.")

    mu = material.shear_modulus()
    kap = material.kappa()
    opening = displacement.delta_uy()
    sif = (2.0 * mu / (kap + 1.0)) * opening * math.sqrt(2.0 * math.pi / displacement.r)
    return ModeIResult(sif=sif, displacement=displacement, material=material)


def compute_mode_i_from_cod(Eprime: float, cod: float, r: float) -> float:
    """Compute K_I directly from COD using K_I = (E'/4) * COD * sqrt(2*pi/r)."""
    if r <= 0:
        raise ValueError("Radial distance r must be positive for DCM.")
    return (Eprime / 4.0) * cod * math.sqrt(2.0 * math.pi / r)


def estimate_plateau_ki(
    records: Iterable[CrackFaceDisplacement],
    material: Material,
    use_median: bool = True,
) -> dict:
    """
    Programmatic helper for stochastic/postprocessing workflows.

    Parameters
    ----------
    records:
        Iterable of crack-face displacement samples.
    material:
        Material properties for DCM conversion.
    use_median:
        True -> robust median plateau, False -> mean plateau.
    """
    vals: list[float] = []
    rs: list[float] = []
    cods: list[float] = []
    samples: list[dict] = []
    for rec in records:
        out = compute_mode_i_sif(rec, material)
        sif = float(out.sif)
        cod = float(rec.delta_uy())
        vals.append(sif)
        rs.append(float(rec.r))
        cods.append(cod)
        samples.append(
            {
                "r": float(rec.r),
                "uy_upper": float(rec.uy_upper),
                "uy_lower": float(rec.uy_lower),
                "cod": cod,
                "KI": sif,
            }
        )
    if not vals:
        raise ValueError("No displacement records provided for DCM plateau estimation.")
    arr = np.asarray(vals, dtype=float)
    r_arr = np.asarray(rs, dtype=float)
    cod_arr = np.asarray(cods, dtype=float)

    # Robust multi-point fit:
    # cod(r) = m * sqrt(r), and KI = [2*mu/(kappa+1)] * m * sqrt(2*pi)
    x = np.sqrt(r_arr)
    y = cod_arr
    c_ki = (2.0 * material.shear_modulus() / (material.kappa() + 1.0)) * math.sqrt(2.0 * math.pi)

    # Inlier filtering based on pointwise KI residuals (median/MAD), then
    # through-origin least-squares fit on cod-vs-sqrt(r).
    if use_median:
        center = float(np.median(arr))
    else:
        center = float(np.mean(arr))
    resid = arr - center
    mad = float(np.median(np.abs(resid)))
    if mad > 0.0:
        inlier_mask = np.abs(resid) <= 3.0 * 1.4826 * mad
    else:
        inlier_mask = np.ones_like(arr, dtype=bool)
    if int(np.count_nonzero(inlier_mask)) < 3:
        inlier_mask = np.ones_like(arr, dtype=bool)

    x_in = x[inlier_mask]
    y_in = y[inlier_mask]
    denom = float(np.dot(x_in, x_in))
    if denom <= 0.0:
        raise ValueError("Invalid DCM fit: zero denominator in cod-vs-sqrt(r) regression.")
    slope = float(np.dot(x_in, y_in) / denom)
    ki_fit = float(c_ki * slope)

    y_hat = slope * x_in
    sse = float(np.sum((y_in - y_hat) ** 2))
    sst = float(np.sum((y_in - np.mean(y_in)) ** 2))
    fit_r2 = float(1.0 - sse / sst) if sst > 0.0 else 1.0

    return {
        "KI_ref": ki_fit,
        "KI_mean": float(np.mean(arr)),
        "KI_std": float(np.std(arr)),
        "n_samples": int(arr.size),
        "n_inliers": int(np.count_nonzero(inlier_mask)),
        "r_min": float(np.min(rs)),
        "r_max": float(np.max(rs)),
        "fit_model": "cod_vs_sqrt_r_through_origin",
        "fit_slope_cod_per_sqrtm": slope,
        "fit_r2": fit_r2,
        "KI_pointwise_median": float(np.median(arr)),
        "KI_pointwise_mean": float(np.mean(arr)),
        "samples": samples,
    }


def _format_result(result: ModeIResult) -> str:
    unit = "MPa*sqrt(m)"
    sif_mpa = result.sif / 1e6
    return (
        f"r = {result.displacement.r:.4e} m | Δuy = {result.displacement.delta_uy():.4e} m | "
        f"K_I = {sif_mpa:.3f} {unit}"
    )


# -----------------------------
# Simple CSV (r_m, uy_upper_m, uy_lower_m)
# -----------------------------
def _read_displacements(path: Path) -> List[CrackFaceDisplacement]:
    """
    Load crack-face displacements from a CSV file.

    CSV must include headers:
      r_m, uy_upper_m, uy_lower_m
    """
    records: List[CrackFaceDisplacement] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                r = float(row["r_m"])
                uy_upper = float(row["uy_upper_m"])
                uy_lower = float(row["uy_lower_m"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("CSV must include numeric columns r_m, uy_upper_m, uy_lower_m") from exc
            records.append(CrackFaceDisplacement(r=r, uy_upper=uy_upper, uy_lower=uy_lower))
    return records


# -----------------------------
# Argument parsing
# -----------------------------
def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Mode I SIF via DCM (interactive if missing args).")

    # Material (optional -> interactive prompts if missing)
    parser.add_argument("--material-E", type=float, required=False, help="Young's modulus in Pa (e.g., 7.31e10)")
    parser.add_argument("--material-nu", type=float, required=False, help="Poisson ratio (e.g., 0.33)")
    parser.add_argument("--plane-strain", action="store_true", help="Use plane strain (default plane stress)")

    # Direct entry
    direct = parser.add_argument_group("Direct entry")
    direct.add_argument("--r", type=float, help="Radial distance r in meters")
    direct.add_argument("--uy-upper", type=float, help="Upper face normal displacement (m)")
    direct.add_argument("--uy-lower", type=float, help="Lower face normal displacement (m)")
    parser.add_argument("--csv", type=Path, help="CSV with columns r_m, uy_upper_m, uy_lower_m")

    # NX node pairing
    nx = parser.add_argument_group("NX node pairing")
    nx.add_argument("--nx-csv", type=Path, help="NX CSV (e.g., Node ID, X Coord, Y Coord, Y)")
    nx.add_argument("--x-tip", type=float, help="Crack tip x-coordinate (NX units). Optional if --auto-tip.")
    nx.add_argument("--y-tip", type=float, help="Crack tip y-coordinate / crack line (NX units). Optional if --auto-tip.")
    nx.add_argument("--nx-units", choices=["mm", "m"], default="mm",
                    help="Units used in NX export for x,y,uy. Default: mm (converted to m internally).")
    nx.add_argument("--y-band", type=float, default=None,
                    help="Half-height band about crack line for face selection (NX units)")
    nx.add_argument("--x-match-tol", type=float, default=None,
                    help="Maximum |x_upper - x_lower| for pairing (NX units)")
    nx.add_argument("--r-min", type=float, default=None,
                    help="_toggle_ manual r-min. If omitted, auto-window selection is used.")
    nx.add_argument("--r-max", type=float, default=None,
                    help="toggle manual r-max. If omitted, auto-window selection is used.")
    nx.add_argument("--plot", action="store_true", help="Plot K_I vs r to identify plateau")

    # Auto-tip controls
    nx.add_argument("--auto-tip", action="store_true",
                    help="Auto-detect y_tip and x_tip from the NX export (if not provided).")
    nx.add_argument("--auto-y-bin", type=float, default=None,
                    help=("Y bin size (NX units) for auto-detecting y_tip from densest y-cluster. "
                          "If omitted, a conservative default is used based on nx-units."))

    # Auto r-window selection controls
    nx.add_argument("--auto-r-window", action="store_true",
                    help="Auto-select r_min/r_max from data even if user provided them (overrides).")
    nx.add_argument("--auto-window-frac", type=float, default=0.35,
                    help="Fraction of points used per sliding window when auto-selecting r-range (default 0.35).")
    nx.add_argument("--auto-window-minpts", type=int, default=8,
                    help="Minimum points per window when auto-selecting r-range (default 8).")

    return parser.parse_args(argv)


def _validate_direct_args(args: argparse.Namespace) -> Optional[CrackFaceDisplacement]:
    provided = [args.r, args.uy_upper, args.uy_lower]
    if any(v is not None for v in provided):
        if not all(v is not None for v in provided):
            raise SystemExit("If using direct entry, --r, --uy-upper, and --uy-lower must all be set.")
        return CrackFaceDisplacement(r=args.r, uy_upper=args.uy_upper, uy_lower=args.uy_lower)
    return None


# -----------------------------
# NX CSV loading with robust mapping + units conversion
# -----------------------------
def _load_nodes(csv_path: Path, nx_units: str) -> "pd.DataFrame":
    import pandas as pd

    df_raw = pd.read_csv(csv_path)
    norm = {c.strip().lower(): c for c in df_raw.columns}

    def pick(*cands: str) -> str:
        for c in cands:
            if c in norm:
                return norm[c]
        raise ValueError(f"Missing one of {cands}. Found columns: {list(df_raw.columns)}")

    col_node = pick("node id", "node_id", "nid", "node")
    col_x = pick("x coord", "x", "x coordinate", "xcoord")
    col_y = pick("y coord", "y coordinate", "ycoord")
    col_uy = pick("uy", "u_y", "y", "displacement y")

    out = pd.DataFrame(
        {
            "node_id": df_raw[col_node].astype(int),
            "x": df_raw[col_x].astype(float),
            "y": df_raw[col_y].astype(float),
            "uy": df_raw[col_uy].astype(float),
        }
    )

    if nx_units == "mm":
        mm_to_m = 1e-3
        out["x"] *= mm_to_m
        out["y"] *= mm_to_m
        out["uy"] *= mm_to_m

    return out


# -----------------------------
# Auto-detect helpers (y_tip and x_tip)
# -----------------------------
def _infer_y_tip(df: "pd.DataFrame", y_band: float, y_bin: float) -> float:
    import numpy as np

    if df.empty:
        raise ValueError("Cannot infer y_tip from empty dataframe.")

    ys = df["y"].to_numpy(dtype=float)
    yq = np.round(ys / y_bin) * y_bin
    vals, counts = np.unique(yq, return_counts=True)
    y_tip = float(vals[int(np.argmax(counts))])

    near = np.abs(ys - y_tip) <= y_band
    logging.info("Auto y_tip: densest y-bin center = %.6g (near-band count=%d)", y_tip, int(np.sum(near)))
    return y_tip


def _infer_x_tip_from_band(df: "pd.DataFrame", y_tip: float, y_band: float) -> float:
    import numpy as np

    band = df[np.abs(df["y"] - y_tip) <= y_band].copy()
    if band.empty:
        raise ValueError("Cannot infer x_tip: no nodes found within y-band around inferred y_tip.")
    x_tip = float(np.max(band["x"].to_numpy(dtype=float)))
    logging.info("Auto x_tip: max x in y-band = %.6g", x_tip)
    return x_tip


# -----------------------------
# Auto r-window selection (from K(r) itself)
# -----------------------------
def _auto_r_window_from_K(paired: "pd.DataFrame", r_col: str = "r", k_col: str = "KI",
                          frac_window: float = 0.35, min_points: int = 8) -> tuple[float, float]:
    """
    Choose r_min and r_max from the data by selecting the contiguous window where K(r)
    is flattest (minimum slope + low scatter) on a sliding linear fit.

    Returns (r_min, r_max) in the same units as r_col (meters internally).
    """
    import numpy as np

    p = paired[[r_col, k_col]].dropna().sort_values(r_col).reset_index(drop=True)
    r = p[r_col].to_numpy(dtype=float)
    k = p[k_col].to_numpy(dtype=float)

    n = len(r)
    if n < max(min_points, 5):
        raise ValueError("Not enough paired points to auto-select r window.")

    w = int(max(min_points, round(frac_window * n)))
    w = min(w, n)

    best_i = 0
    best_score = None

    r_span = float(r[-1] - r[0]) if n > 1 else 1.0

    for i in range(0, n - w + 1):
        rr = r[i:i + w]
        kk = k[i:i + w]
        A = np.vstack([rr, np.ones_like(rr)]).T
        slope, _ = np.linalg.lstsq(A, kk, rcond=None)[0]

        k_med = float(np.median(kk))
        scatter = float(np.std(kk) / (abs(k_med) + 1e-30))

        # Dimensionless-ish slope metric + scatter penalty
        slope_norm = abs(float(slope)) * r_span / (abs(k_med) + 1e-30)
        score = slope_norm + 0.5 * scatter

        if best_score is None or score < best_score:
            best_score = score
            best_i = i

    r_min = float(r[best_i])
    r_max = float(r[best_i + w - 1])
    return r_min, r_max


# -----------------------------
# Pairing logic
# -----------------------------
def _pair_faces_by_x(df: "pd.DataFrame", x_tip: float, y_tip: float, y_band: float, x_match_tol: float) -> "pd.DataFrame":
    import numpy as np

    band = df[np.abs(df["y"] - y_tip) <= y_band].copy()
    upper = band[band["y"] > y_tip].copy()
    lower = band[band["y"] < y_tip].copy()

    logging.info("Nodes in band: %s | upper: %s | lower: %s", len(band), len(upper), len(lower))

    upper.sort_values("x", inplace=True)
    lower.sort_values("x", inplace=True)

    lower_x = lower["x"].to_numpy()
    lower_idx = lower.index.to_numpy()

    pairs = []
    for _, rowu in upper.iterrows():
        xu = float(rowu["x"])
        j = int(np.searchsorted(lower_x, xu))
        candidates = []
        if 0 <= j < len(lower_x):
            candidates.append(j)
        if 0 <= j - 1 < len(lower_x):
            candidates.append(j - 1)

        best = None
        best_dx = None
        for c in candidates:
            dx = abs(float(lower_x[c]) - xu)
            if best is None or dx < float(best_dx):
                best = c
                best_dx = dx

        if best is not None and best_dx is not None and best_dx <= x_match_tol:
            il = lower_idx[best]
            rowl = lower.loc[il]
            pairs.append(
                {
                    "x": 0.5 * (float(rowu["x"]) + float(rowl["x"])),
                    "y_upper": float(rowu["y"]),
                    "y_lower": float(rowl["y"]),
                    "uy_upper": float(rowu["uy"]),
                    "uy_lower": float(rowl["uy"]),
                    "node_upper": int(rowu.get("node_id", rowu.name)),
                    "node_lower": int(rowl.get("node_id", rowl.name)),
                    "dx_match": float(best_dx),
                }
            )

    paired = df.__class__(pairs)
    logging.info("Paired node rows: %s", len(paired))
    return paired


def _process_nx_nodes(args: argparse.Namespace, material: Material) -> int:
    import numpy as np

    if args.nx_csv is None:
        return 0

    if not args.nx_csv.exists():
        raise SystemExit(f"NX CSV not found: {args.nx_csv}")

    # Prompt for missing tolerances (needed for pairing + auto-tip)
    if args.y_band is None:
        args.y_band = _prompt_float("y-band (half-height band about crack line)", default=0.05)
    if args.x_match_tol is None:
        args.x_match_tol = _prompt_float("x-match-tol (pairing tolerance)", default=0.05)

    # Load nodes (converted to meters if nx_units == mm)
    df = _load_nodes(args.nx_csv, nx_units=args.nx_units)

    # Convert tolerances + optional x_tip/y_tip to meters if user entered in mm
    if args.nx_units == "mm":
        mm_to_m = 1e-3
        args.y_band *= mm_to_m
        args.x_match_tol *= mm_to_m
        if args.x_tip is not None:
            args.x_tip *= mm_to_m
        if args.y_tip is not None:
            args.y_tip *= mm_to_m
        if args.auto_y_bin is not None:
            args.auto_y_bin *= mm_to_m

    # Auto-tip (fills missing; does not override user-provided)
    if args.auto_tip:
        if args.auto_y_bin is None:
            # Default y-bin in meters: 0.01 mm -> 1e-5 m
            args.auto_y_bin = 1e-5

        if args.y_tip is None:
            args.y_tip = _infer_y_tip(df, y_band=float(args.y_band), y_bin=float(args.auto_y_bin))
        if args.x_tip is None:
            args.x_tip = _infer_x_tip_from_band(df, y_tip=float(args.y_tip), y_band=float(args.y_band))

        logging.info("Using (auto) y_tip = %.6g m, x_tip = %.6g m", float(args.y_tip), float(args.x_tip))

    # If not auto, prompt for missing tips
    if args.x_tip is None:
        args.x_tip = _prompt_float("Crack tip x-coordinate (NX units)")
        if args.nx_units == "mm":
            args.x_tip *= 1e-3
    if args.y_tip is None:
        args.y_tip = _prompt_float("Crack tip y-coordinate (NX units)")
        if args.nx_units == "mm":
            args.y_tip *= 1e-3

    # Only consider nodes behind the tip
    df = df[df["x"] <= float(args.x_tip)].copy()

    paired = _pair_faces_by_x(df, float(args.x_tip), float(args.y_tip), float(args.y_band), float(args.x_match_tol))
    if paired.empty:
        raise SystemExit("No upper/lower face node pairs found. Adjust y-band or x-match-tol.")

    # Compute r, COD, KI for all available pairs (no r-filter yet)
    paired["r"] = (float(args.x_tip) - paired["x"]).astype(float)
    paired = paired[paired["r"] > 0].copy()

    Eprime = material.effective_modulus()
    paired["cod"] = paired["uy_upper"] - paired["uy_lower"]
    paired["KI"] = paired.apply(lambda row: compute_mode_i_from_cod(Eprime, row["cod"], row["r"]), axis=1)
    paired.sort_values("r", inplace=True)

    # --- Auto-select r-window if user did not provide it, or if --auto-r-window set ---
    do_auto_window = args.auto_r_window or (args.r_min is None) or (args.r_max is None)

    # Convert user-provided r-min/r-max to meters if nx_units == mm (they're in NX units)
    # But only if they exist and auto-window won't override them.
    if args.nx_units == "mm" and not args.auto_r_window:
        if args.r_min is not None:
            args.r_min *= 1e-3
        if args.r_max is not None:
            args.r_max *= 1e-3

    if do_auto_window:
        rmin_auto, rmax_auto = _auto_r_window_from_K(
            paired,
            frac_window=float(args.auto_window_frac),
            min_points=int(args.auto_window_minpts),
        )
        logging.info("Auto-selected r-window from data: r_min=%.6g m, r_max=%.6g m", rmin_auto, rmax_auto)
        args.r_min = rmin_auto
        args.r_max = rmax_auto
    else:
        # If user specified both, enforce them
        if args.r_min is None or args.r_max is None:
            raise SystemExit("Internal error: expected r_min and r_max to be set here.")

    # Apply r-window
    paired_win = paired[(paired["r"] >= float(args.r_min)) & (paired["r"] <= float(args.r_max))].copy()
    if paired_win.empty:
        raise SystemExit("No pairs remain after r-range filtering. Auto-window failed or r-range too tight.")

    logging.info("Sample (r, COD, K_I) within chosen window:\n%s", paired_win[["r", "cod", "KI", "dx_match"]].head(10))

    # Plot (optionally annotate window)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(paired["r"], paired["KI"] / 1e6, marker="o")
        # plt.axvline(float(args.r_min), linestyle="--")
        # plt.axvline(float(args.r_max), linestyle="--")
        plt.xlabel("r (m)  [distance behind tip]")
        plt.ylabel("K_I (MPa*sqrt(m))")
        plt.grid(True)
        plt.title("DCM Mode I from COD")
        plt.tight_layout()

        plot_path = args.nx_csv.with_name(f"dcm_{args.nx_csv.stem}_KI_vs_r.png")
        plt.savefig(plot_path, dpi=200)
        logging.info("Saved plot to %s", plot_path)

    # Plateau estimate on the chosen window: median (robust)
    KI_est = float(np.median(paired_win["KI"]))
    logging.info(
        "Plateau estimate (median over auto-window): K_I ≈ %.6g Pa*sqrt(m) (%.3f MPa*sqrt(m))",
        KI_est,
        KI_est / 1e6,
    )

    # Output full paired file + chosen window file
    out_all = args.nx_csv.with_name(f"dcm_{args.nx_csv.stem}_pairs_and_KI_all.csv")
    paired.to_csv(out_all, index=False)
    logging.info("Wrote full paired results to %s", out_all)

    out_win = args.nx_csv.with_name(f"dcm_{args.nx_csv.stem}_pairs_and_KI_window.csv")
    paired_win.to_csv(out_win, index=False)
    logging.info("Wrote windowed results to %s", out_win)

    return 0


# -----------------------------
# Main
# -----------------------------
def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    args = _parse_args(argv)

    # ---- Material prompts (Al defaults) ----
    if args.material_E is None or args.material_nu is None:
        print("\nMaterial properties (press Enter to accept defaults for Al):")
    E = args.material_E if args.material_E is not None else _prompt_float("Young's modulus E [Pa]", default=7.31e10)
    nu = args.material_nu if args.material_nu is not None else _prompt_float("Poisson ratio nu", default=0.33)

    material = Material(elastic_modulus=E, poisson_ratio=nu, plane_strain=args.plane_strain)

    # ---- Choose workflow (interactive if nothing specified) ----
    if args.nx_csv is None and args.csv is None and _validate_direct_args(args) is None:
        print("\nChoose input mode:")
        print("  1) NX node pairing (recommended)")
        print("  2) Simple CSV (r_m, uy_upper_m, uy_lower_m)")
        print("  3) Direct entry (single r, uy_upper, uy_lower)")
        choice = _prompt_str("Mode", default="1")

        if choice == "1":
            p = _prompt_str("Path to NX CSV file (export from NX)")
            args.nx_csv = Path(p)
            args.nx_units = _prompt_str("NX units (mm or m)", default="mm").lower()
            args.auto_tip = _prompt_bool("Auto-detect crack tip (x_tip/y_tip) from CSV?", default=True)
            args.plot = _prompt_bool("Plot K_I vs r?", default=True)

            # Auto-window by default in interactive mode
            args.auto_r_window = _prompt_bool("Auto-select r_min/r_max from K(r)?", default=True)

            # If user turns off auto-window, allow manual entry
            if not args.auto_r_window:
                args.r_min = _prompt_float("r-min (NX units)")
                args.r_max = _prompt_float("r-max (NX units)")

        elif choice == "2":
            p = _prompt_str("Path to CSV with r_m, uy_upper_m, uy_lower_m")
            args.csv = Path(p)

        elif choice == "3":
            args.r = _prompt_float("r (meters)")
            args.uy_upper = _prompt_float("uy_upper (meters)")
            args.uy_lower = _prompt_float("uy_lower (meters)")
        else:
            raise SystemExit("Unknown mode selection.")

    # ---- NX mode ----
    if args.nx_csv:
        return _process_nx_nodes(args, material)

    # ---- Direct / simple CSV modes ----
    displacements: List[CrackFaceDisplacement] = []

    direct_entry = _validate_direct_args(args)
    if direct_entry:
        displacements.append(direct_entry)

    if args.csv:
        if not args.csv.exists():
            raise SystemExit(f"CSV file not found: {args.csv}")
        displacements.extend(_read_displacements(args.csv))

    if not displacements:
        raise SystemExit("No displacement data provided. Use NX mode, --csv, or direct entry flags.")

    for displacement in displacements:
        result = compute_mode_i_sif(displacement, material)
        print(_format_result(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

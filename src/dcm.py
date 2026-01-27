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

NEW in this version:
- Interactive prompts: you can run `./src/dcm.py` and it will ask for missing inputs.
- Robust NX column mapping (handles "Node ID", "X Coord", "Y Coord", "Y").
- Units handling for NX exports (mm or m). Recommended: export in mm and set --nx-units mm
  (default), and the script converts to meters internally so K is in MPa*sqrt(m).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

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
    """Compute K_I directly from COD using the classical E'/8 relation."""
    if r <= 0:
        raise ValueError("Radial distance r must be positive for DCM.")
    return (Eprime / 8.0) * cod * math.sqrt(2.0 * math.pi / r)


def _format_result(result: ModeIResult) -> str:
    # If everything is SI (m, Pa), this prints MPa*sqrt(m) correctly.
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

    # Material (now optional -> interactive prompts if missing)
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
    nx.add_argument("--x-tip", type=float, help="Crack tip x-coordinate (model units)")
    nx.add_argument("--y-tip", type=float, help="Crack tip y-coordinate (model units)")
    nx.add_argument("--nx-units", choices=["mm", "m"], default="mm",
                    help="Units used in NX export for x,y,uy. Default: mm (converted to m internally).")
    nx.add_argument("--y-band", type=float, default=None,
                    help="Half-height band about crack line for face selection (model units)")
    nx.add_argument("--x-match-tol", type=float, default=None,
                    help="Maximum |x_upper - x_lower| for pairing (model units)")
    nx.add_argument("--r-min", type=float, default=None, help="Minimum r to include (model units)")
    nx.add_argument("--r-max", type=float, default=None, help="Maximum r to include (model units)")
    nx.add_argument("--plot", action="store_true", help="Plot K_I vs r to identify plateau")

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

    # Handle your NX export headers
    col_node = pick("node id", "node_id", "nid", "node")
    col_x    = pick("x coord", "x", "x coordinate", "xcoord")
    col_y    = pick("y coord", "y coordinate", "ycoord")
    # Displacement Uy is often labeled just "Y" in NX exports
    col_uy   = pick("uy", "u_y", "y", "displacement y")

    out = pd.DataFrame(
        {
            "node_id": df_raw[col_node].astype(int),
            "x": df_raw[col_x].astype(float),
            "y": df_raw[col_y].astype(float),
            "uy": df_raw[col_uy].astype(float),
        }
    )

    # Convert to meters if needed (recommended for correct MPa*sqrt(m))
    if nx_units == "mm":
        mm_to_m = 1e-3
        out["x"] *= mm_to_m
        out["y"] *= mm_to_m
        out["uy"] *= mm_to_m

    return out


def _pair_faces_by_x(df: "pd.DataFrame", x_tip: float, y_tip: float, y_band: float, x_match_tol: float) -> "pd.DataFrame":
    import numpy as np

    # Select nodes within a band around the crack line (y ≈ y_tip)
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

    paired = df.__class__(pairs)  # pandas DataFrame
    logging.info("Paired node rows: %s", len(paired))
    return paired


def _process_nx_nodes(args: argparse.Namespace, material: Material) -> int:
    import numpy as np

    if args.nx_csv is None:
        return 0

    if not args.nx_csv.exists():
        raise SystemExit(f"NX CSV not found: {args.nx_csv}")

    # Prompt for missing NX params
    if args.x_tip is None:
        args.x_tip = _prompt_float("Crack tip x-coordinate (NX units)")
    if args.y_tip is None:
        args.y_tip = _prompt_float("Crack tip y-coordinate (NX units)")

    # Default band/tols depend on units
    # If nx_units == mm, we convert to meters, so defaults should be in meters too.
    if args.y_band is None:
        args.y_band = _prompt_float("y-band (half-height band about crack line)", default=0.05)
    if args.x_match_tol is None:
        args.x_match_tol = _prompt_float("x-match-tol (pairing tolerance)", default=0.05)
    if args.r_min is None:
        args.r_min = _prompt_float("r-min (distance behind tip)", default=0.2)
    if args.r_max is None:
        args.r_max = _prompt_float("r-max (distance behind tip)", default=2.0)

    # If user said NX units are mm, convert all those inputs from mm->m to match converted data
    if args.nx_units == "mm":
        mm_to_m = 1e-3
        args.x_tip *= mm_to_m
        args.y_tip *= mm_to_m
        args.y_band *= mm_to_m
        args.x_match_tol *= mm_to_m
        args.r_min *= mm_to_m
        args.r_max *= mm_to_m

    df = _load_nodes(args.nx_csv, nx_units=args.nx_units)

    # Only consider nodes behind the tip (x <= x_tip) for r = x_tip - x
    df = df[df["x"] <= args.x_tip].copy()

    paired = _pair_faces_by_x(df, args.x_tip, args.y_tip, args.y_band, args.x_match_tol)
    if paired.empty:
        raise SystemExit("No upper/lower face node pairs found. Adjust y-band or x-match-tol.")

    paired["r"] = (args.x_tip - paired["x"]).astype(float)
    paired = paired[(paired["r"] >= args.r_min) & (paired["r"] <= args.r_max)].copy()
    if paired.empty:
        raise SystemExit("No pairs remain after r-range filtering. Adjust r-min/r-max.")

    Eprime = material.effective_modulus()
    paired["cod"] = paired["uy_upper"] - paired["uy_lower"]
    paired["KI"] = paired.apply(lambda row: compute_mode_i_from_cod(Eprime, row["cod"], row["r"]), axis=1)

    paired.sort_values("r", inplace=True)
    logging.info("Sample (r, COD, K_I):\n%s", paired[["r", "cod", "KI", "dx_match"]].head(10))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(paired["r"], paired["KI"] / 1e6, marker="o")
        plt.xlabel("r (m)  [distance behind tip]")
        plt.ylabel("K_I (MPa*sqrt(m))")
        plt.grid(True)
        plt.title("DCM Mode I from COD")
        plt.tight_layout()

        plot_path = args.nx_csv.with_name(f"dcm_{args.nx_csv.stem}_KI_vs_r.png")
        plt.savefig(plot_path, dpi=200)
        logging.info("Saved plot to %s", plot_path)


    # Plateau estimate: median of middle 50%
    n = len(paired)
    lo = int(0.25 * n)
    hi = int(0.75 * n)
    KI_est = float(np.median(paired["KI"].iloc[lo:hi]))
    logging.info("Plateau estimate (median middle 50%%): K_I ≈ %.6g Pa*sqrt(m) (%.3f MPa*sqrt(m))", KI_est, KI_est / 1e6)

    # Output file (avoid overwriting across crack lengths)
    out_path = args.nx_csv.with_name(f"dcm_{args.nx_csv.stem}_pairs_and_KI.csv")
    paired.to_csv(out_path, index=False)
    logging.info("Wrote paired results to %s", out_path)

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
            args.plot = _prompt_bool("Plot K_I vs r?", default=True)

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

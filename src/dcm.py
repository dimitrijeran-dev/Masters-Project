"""DCM utilities for Mode I stress intensity factor estimation.

This version supports two workflows:

1) Direct Mode I calculation from crack-face displacements at known radial
   locations (CSV or CLI flags).
2) Automatic pairing of upper/lower crack-face nodes exported from Siemens NX
   (or similar) followed by a radial sweep to estimate the K_I plateau.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    import pandas as pd

@dataclass(frozen=True)
class CrackFaceDisplacement:
    """Crack-face displacement pair at a radial distance from the crack tip."""

    r: float
    uy_upper: float
    uy_lower: float

    def delta_uy(self) -> float:
        """Return the displacement opening (upper minus lower)."""

        return self.uy_upper - self.uy_lower


@dataclass(frozen=True)
class Material:
    """Material properties needed for DCM calculations."""

    elastic_modulus: float  # Young's modulus (Pa)
    poisson_ratio: float  # Poisson's ratio (dimensionless)
    plane_strain: bool = False

    def shear_modulus(self) -> float:
        """Return the shear modulus (μ = E / 2(1 + ν))."""

        return self.elastic_modulus / (2 * (1 + self.poisson_ratio))

    def kappa(self) -> float:
        """Return the Kolosov constant (κ) for plane stress or plane strain."""

        if self.plane_strain:
            return 3 - 4 * self.poisson_ratio
        return (3 - self.poisson_ratio) / (1 + self.poisson_ratio)

    def effective_modulus(self) -> float:
        """Return E' (E for plane stress, E/(1-ν²) for plane strain)."""

        if self.plane_strain:
            return self.elastic_modulus / (1 - self.poisson_ratio**2)
        return self.elastic_modulus


@dataclass(frozen=True)
class ModeIResult:
    """Container for Mode I stress intensity factor calculations."""

    sif: float
    displacement: CrackFaceDisplacement
    material: Material


def compute_mode_i_sif(displacement: CrackFaceDisplacement, material: Material) -> ModeIResult:
    """Compute the Mode I stress intensity factor using the DCM relation.

    K_I = [2 μ / (κ + 1)] * Δu_y * sqrt(2π / r)
    """

    if displacement.r <= 0:
        raise ValueError("Radial distance r must be positive for DCM.")

    mu = material.shear_modulus()
    kappa = material.kappa()
    opening = displacement.delta_uy()
    sif = (2 * mu / (kappa + 1.0)) * opening * math.sqrt(2 * math.pi / displacement.r)

    return ModeIResult(sif=sif, displacement=displacement, material=material)


def compute_mode_i_from_cod(Eprime: float, cod: float, r: float) -> float:
    """Compute K_I directly from COD using the classical E'/8 relation."""

    if r <= 0:
        raise ValueError("Radial distance r must be positive for DCM.")
    return (Eprime / 8.0) * cod * math.sqrt(2.0 * math.pi / r)


def _read_displacements(path: Path) -> List[CrackFaceDisplacement]:
    """Load crack-face displacements from a CSV file.

    The CSV must include headers: ``r_m``, ``uy_upper_m``, ``uy_lower_m``.
    Additional columns are ignored.
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
                raise ValueError(
                    "CSV must include numeric columns r_m, uy_upper_m, uy_lower_m"
                ) from exc
            records.append(CrackFaceDisplacement(r=r, uy_upper=uy_upper, uy_lower=uy_lower))
    return records


def _format_result(result: ModeIResult) -> str:
    unit = "MPa*sqrt(m)"
    sif_mpa = result.sif / 1e6
    return (
        f"r = {result.displacement.r:.4e} m | Δuy = {result.displacement.delta_uy():.4e} m | "
        f"K_I = {sif_mpa:.3f} {unit}"
    )


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Mode I SIF via DCM.")
    parser.add_argument(
        "--material-E",
        type=float,
        required=True,
        help="Young's modulus in Pascals (e.g., 7.0e10 for 70 GPa)",
    )
    parser.add_argument(
        "--material-nu",
        type=float,
        required=True,
        help="Poisson ratio (e.g., 0.33)",
    )
    parser.add_argument(
        "--plane-strain",
        action="store_true",
        help="Use plane strain (default is plane stress)",
    )

    direct = parser.add_argument_group("Direct entry")
    direct.add_argument("--r", type=float, help="Radial distance r in meters")
    direct.add_argument(
        "--uy-upper",
        type=float,
        help="Normal displacement on upper crack face in meters",
    )
    direct.add_argument(
        "--uy-lower",
        type=float,
        help="Normal displacement on lower crack face in meters",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="CSV file with columns r_m, uy_upper_m, uy_lower_m",
    )

    nx_group = parser.add_argument_group("NX node pairing")
    nx_group.add_argument(
        "--nx-csv",
        type=Path,
        help="CSV with columns x, y, uy (ux optional, node_id optional)",
    )
    nx_group.add_argument("--x-tip", type=float, help="Crack tip x-coordinate (model units)")
    nx_group.add_argument("--y-tip", type=float, help="Crack tip y-coordinate (model units)")
    nx_group.add_argument(
        "--y-band",
        type=float,
        default=0.02,
        help="Half-height band about crack line for face selection (model units)",
    )
    nx_group.add_argument(
        "--x-match-tol",
        type=float,
        default=0.02,
        help="Maximum |x_upper - x_lower| when pairing faces (model units)",
    )
    nx_group.add_argument(
        "--r-min",
        type=float,
        default=None,
        help="Minimum r (x_tip - x) to include (model units)",
    )
    nx_group.add_argument(
        "--r-max",
        type=float,
        default=None,
        help="Maximum r (x_tip - x) to include (model units)",
    )
    nx_group.add_argument(
        "--plot",
        action="store_true",
        help="Plot K_I vs r for visual plateau identification",
    )

    return parser.parse_args(argv)


def _validate_direct_args(args: argparse.Namespace) -> Optional[CrackFaceDisplacement]:
    provided = [args.r, args.uy_upper, args.uy_lower]
    if any(value is not None for value in provided):
        if not all(value is not None for value in provided):
            raise SystemExit("If using direct entry, r, uy-upper, and uy-lower must all be set.")
        return CrackFaceDisplacement(r=args.r, uy_upper=args.uy_upper, uy_lower=args.uy_lower)
    return None


def _load_nodes(csv_path: Path) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_path)
    required = {"x", "y", "uy"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"CSV missing required column(s): {', '.join(sorted(missing))}")

    # Preserve original casing but normalize for lookup
    cols = {c.lower(): c for c in df.columns}
    result = pd.DataFrame(
        {
            "node_id": df[cols.get("node_id", None)] if "node_id" in cols else np.arange(len(df)),
            "x": df[cols["x"]].astype(float),
            "y": df[cols["y"]].astype(float),
            "uy": df[cols["uy"]].astype(float),
        }
    )
    if "ux" in cols:
        result["ux"] = df[cols["ux"]].astype(float)
    return result


def _pair_faces_by_x(
    df: pd.DataFrame, x_tip: float, y_tip: float, y_band: float, x_match_tol: float
) -> pd.DataFrame:
    import numpy as np

    band = df[np.abs(df["y"] - y_tip) <= y_band].copy()
    upper = band[band["y"] > y_tip].copy()
    lower = band[band["y"] < y_tip].copy()

    logging.info(
        "Nodes in band: %s | upper: %s | lower: %s", len(band), len(upper), len(lower)
    )

    upper.sort_values("x", inplace=True)
    lower.sort_values("x", inplace=True)

    lower_x = lower["x"].to_numpy()
    lower_idx = lower.index.to_numpy()

    pairs = []
    for _, rowu in upper.iterrows():
        xu = rowu["x"]
        j = np.searchsorted(lower_x, xu)
        candidates = []
        if 0 <= j < len(lower_x):
            candidates.append(j)
        if 0 <= j - 1 < len(lower_x):
            candidates.append(j - 1)

        best = None
        best_dx = None
        for c in candidates:
            dx = abs(lower_x[c] - xu)
            if best is None or dx < best_dx:
                best = c
                best_dx = dx

        if best is not None and best_dx is not None and best_dx <= x_match_tol:
            il = lower_idx[best]
            rowl = lower.loc[il]
            pairs.append(
                {
                    "x": 0.5 * (rowu["x"] + rowl["x"]),
                    "y_upper": rowu["y"],
                    "y_lower": rowl["y"],
                    "uy_upper": rowu["uy"],
                    "uy_lower": rowl["uy"],
                    "node_upper": rowu.get("node_id", rowu.name),
                    "node_lower": rowl.get("node_id", rowl.name),
                    "dx_match": best_dx,
                }
            )

    paired = pd.DataFrame(pairs)
    logging.info("Paired node rows: %s", len(paired))
    return paired


def _process_nx_nodes(args: argparse.Namespace, material: Material) -> int:
    import matplotlib.pyplot as plt
    import numpy as np

    if args.nx_csv is None:
        return 0

    if args.x_tip is None or args.y_tip is None:
        raise SystemExit("--x-tip and --y-tip are required with --nx-csv")

    df = _load_nodes(args.nx_csv)
    df = df[df["x"] <= args.x_tip].copy()
    paired = _pair_faces_by_x(df, args.x_tip, args.y_tip, args.y_band, args.x_match_tol)
    if paired.empty:
        raise SystemExit("No upper/lower face node pairs found. Check y-band or x-match-tol.")

    paired["r"] = (args.x_tip - paired["x"]).astype(float)
    if args.r_min is not None:
        paired = paired[paired["r"] >= args.r_min]
    if args.r_max is not None:
        paired = paired[paired["r"] <= args.r_max]

    if paired.empty:
        raise SystemExit("No pairs remain after r-range filtering. Adjust r-min/r-max.")

    Eprime = material.effective_modulus()
    paired["cod"] = paired["uy_upper"] - paired["uy_lower"]
    paired["KI"] = paired.apply(
        lambda row: compute_mode_i_from_cod(Eprime, row["cod"], row["r"]), axis=1
    )

    paired.sort_values("r", inplace=True)
    logging.info("Sample (r, COD, K_I):\n%s", paired[["r", "cod", "KI", "dx_match"]].head(10))

    if args.plot:
        plt.figure()
        plt.plot(paired["r"], paired["KI"] / 1e6, marker="o")
        plt.xlabel("r (distance behind tip)")
        plt.ylabel("K_I (MPa*sqrt(model length))")
        plt.grid(True)
        plt.title("DCM Mode I from COD")
        plt.tight_layout()
        plt.show()

    n = len(paired)
    lo = int(0.25 * n)
    hi = int(0.75 * n)
    KI_est = float(np.median(paired["KI"].iloc[lo:hi]))
    logging.info("Plateau estimate (median middle 50%%): K_I ≈ %.6g Pa*sqrt(L)", KI_est)

    out_path = args.nx_csv.with_name("dcm_pairs_and_KI.csv")
    paired.to_csv(out_path, index=False)
    logging.info("Wrote paired results to %s", out_path)
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    args = _parse_args(argv)
    material = Material(
        elastic_modulus=args.material_E,
        poisson_ratio=args.material_nu,
        plane_strain=args.plane_strain,
    )

    if args.nx_csv:
        return _process_nx_nodes(args, material)

    displacements: List[CrackFaceDisplacement] = []

    direct_entry = _validate_direct_args(args)
    if direct_entry:
        displacements.append(direct_entry)

    if args.csv:
        if not args.csv.exists():
            raise SystemExit(f"CSV file not found: {args.csv}")
        displacements.extend(_read_displacements(args.csv))

    if not displacements:
        raise SystemExit("No displacement data provided. Use --csv, --nx-csv, or direct entry flags.")

    for displacement in displacements:
        result = compute_mode_i_sif(displacement, material)
        print(_format_result(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

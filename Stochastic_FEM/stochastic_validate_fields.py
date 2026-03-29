#!/usr/bin/env python3
"""
stochastic_validate_fields.py

Upgraded stochastic validation / postprocessing script that mirrors the
behavior of the original validator, while supporting:

- stochastic elementwise material (E_elem in fields*.npz)
- both geometry options:
    * plate_edge_crack
    * plate_hole_edge_crack
- realization selection via realization_id

Outputs
-------
- J_path_independence*.png
- KI_vs_rout*.png
- sigma_yy_and_vonmises_vs_r*.png
- sigma_sqrt_2pir_vs_r*.png
- validation_line*.csv
- validation_summary*.json

Notes
-----
- Uses elementwise E in the J-integral if available.
- Uses scalar E_for_KI = mean(E_elem) when converting J -> KI.
- Stress-line sampling is performed along the crack extension line:
      x = tip_x + r * crack_dir_x
      y = tip_y + r * crack_dir_y
  so it works for both supported geometries.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import matplotlib.pyplot as plt
import meshio
import numpy as np

THIS = Path(__file__).resolve()
ROOT = THIS.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.append(str(ROOT.parent))
if str(ROOT.parent / "src") not in sys.path:
    sys.path.append(str(ROOT.parent / "src"))

from src.configs.run_io import load_runtime_config, update_runtime_config
from src.run_manifest import load_run_manifest, write_run_manifest

# Prefer the stochastic-capable J-integral if present in the same folder.
try:
    from J_Integral_stochastic_modified import (
        sweep_J_rout,
        sweep_interaction_rout,
    )  # type: ignore
except Exception:
    from J_Integral import sweep_J_rout  # type: ignore
    sweep_interaction_rout = None

try:
    from src.dcm import CrackFaceDisplacement, Material, estimate_plateau_ki  # type: ignore
except Exception:
    CrackFaceDisplacement = None  # type: ignore
    Material = None  # type: ignore
    estimate_plateau_ki = None  # type: ignore


@dataclass
class ValConfig:
    run_dir: Path = Path("Data/New Data/40_mm_plate_edge_crack")
    realization_id: Optional[int] = None
    run_all_realizations: bool = True
    realization_glob: str = "fields_mc*.npz"

    # Fallback material / loading info
    E: float = 73.1e9
    nu: float = 0.33
    plane_stress: bool = True
    sigma_nominal: float = 50e6

    # Stress sampling
    r_min: float = 2.0e-4
    r_max: float = 3.0e-2
    n_r: int = 200

    # J sweep
    r_in: float = 0.006
    r_out_list: Tuple[float, ...] = (0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.022, 0.024, 0.026, 0.028, 0.03, 0.032)
    crack_face_exclusion: float = 5.0e-4
    
    # Interaction Integral
    use_interaction_integral_for_stochastic: bool = False
    E_tip_for_aux: Optional[float] = None
    interaction_modes: Tuple[str, ...] = ("I", "II")
    interaction_use_inhomogeneity_correction: bool = True
    interaction_take_abs_KI: bool = True

    # Output control
    export_csv: bool = True
    csv_name: str = "validation_line.csv"
    summary_name: str = "validation_summary.json"
    aggregate_summary_name: str = "validation_summary_all_realizations.json"

    # Optional DCM post-processing from nodal crack-face displacements
    enable_dcm_from_fields: bool = True
    dcm_r_min: float = 2.0e-4
    dcm_r_max: float = 2.0e-3
    dcm_n_bins: int = 48
    dcm_y_band_scale: float = 1.5
    dcm_use_median: bool = True

    # Optional crack-length sweep + lifing comparison
    run_crack_length_sweep: bool = False
    sweep_root: Path = Path("Data/New Data")
    sweep_glob: str = "meshrun_*mm"
    deterministic_glob: str = "meshrun_*mm"
    stochastic_glob: str = "stochastic_*mm"
    lifing_out_dir: Path = Path("Data/Fatigue Outputs/stochastic_lifing")
    life_R: float = 0.1
    paris_C: float = 1.0e-10
    paris_m: float = 3.0
    life_nsamples: int = 2000
    life_C_cov: float = 0.10
    life_m_std: float = 0.10
    life_sigma_scale_mean: float = 1.0
    life_sigma_scale_cov: float = 0.05


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _as_interaction_modes(value: Any, default: Tuple[str, ...]) -> Tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        raw = [value]
    else:
        try:
            raw = list(value)
        except TypeError:
            return default
    out: List[str] = []
    for item in raw:
        token = str(item).strip().upper()
        if token in {"I", "II"} and token not in out:
            out.append(token)
    return tuple(out) if out else default


def crack_tip_xy(meta: dict) -> np.ndarray:
    return np.asarray(meta["tip"], dtype=float).reshape(2)


def crack_start_xy(meta: dict) -> np.ndarray:
    if "crack_start" in meta:
        return np.asarray(meta["crack_start"], dtype=float).reshape(2)
    a = float(meta["a"])
    geom = meta.get("geometry_type", "plate_edge_crack")
    if geom == "plate_edge_crack":
        return np.array([0.0, 0.0], dtype=float)
    if geom == "plate_hole_edge_crack":
        W = float(meta["W"])
        hole_radius = float(meta["hole_radius"])
        cx, cy = meta.get("hole_center", [W / 2.0, 0.0])
        return np.array([float(cx) + hole_radius, float(cy)], dtype=float)
    raise ValueError(f"Unsupported geometry_type={geom!r}")


def crack_dir_xy(meta: dict) -> np.ndarray:
    cd = np.asarray(meta.get("crack_dir", [1.0, 0.0]), dtype=float).reshape(2)
    n = np.linalg.norm(cd)
    if n <= 0:
        raise ValueError("Invalid crack_dir")
    return cd / n

def discover_realization_ids(run_dir: Path, pattern: str = "fields_mc*.npz") -> List[int]:
    ids = []
    for p in sorted(run_dir.glob(pattern)):
        stem = p.stem  # e.g. fields_mc0007
        if "_mc" not in stem:
            continue
        try:
            ids.append(int(stem.split("_mc")[-1]))
        except ValueError:
            continue
    return ids

def load_solution(npz_path: Path):
    data = np.load(npz_path)
    pts = np.asarray(data["pts"], float)
    conn = np.asarray(data["conn"], int)
    u = np.asarray(data["u"], float).reshape(-1)
    E_elem = np.asarray(data["E_elem"], float) if "E_elem" in data else None
    return pts, conn, u, E_elem


def nodal_characteristic_size(pts: np.ndarray, conn: np.ndarray) -> float:
    edge_lengths = []
    for nodes in conn:
        p = pts[nodes, :]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for i, j in edges:
            edge_lengths.append(np.linalg.norm(p[j] - p[i]))
    if not edge_lengths:
        return 1e-6
    return float(np.median(np.asarray(edge_lengths, dtype=float)))


def estimate_dcm_from_fields(
    pts: np.ndarray,
    conn: np.ndarray,
    u: np.ndarray,
    tip: np.ndarray,
    crack_start: np.ndarray,
    cfg: ValConfig,
    E_scalar: float,
    nu: float,
    plane_stress: bool,
) -> Dict[str, Any]:
    if estimate_plateau_ki is None or Material is None or CrackFaceDisplacement is None:
        return {"available": False, "reason": "dcm_import_failed"}

    y_tip = float(tip[1])
    x_tip = float(tip[0])
    crack_len = max(float(np.linalg.norm(tip - crack_start)), 1e-12)
    h_char = nodal_characteristic_size(pts, conn)
    y_band = max(cfg.dcm_y_band_scale * h_char, 5e-6)

    x = pts[:, 0]
    y = pts[:, 1]
    uy = u[1::2]

    upper_mask = (
        (np.abs(y - y_tip) <= y_band)
        & (y >= y_tip)
        & (x <= x_tip - cfg.dcm_r_min)
        & (x >= x_tip - cfg.dcm_r_max)
    )
    lower_mask = (
        (np.abs(y - y_tip) <= y_band)
        & (y <= y_tip)
        & (x <= x_tip - cfg.dcm_r_min)
        & (x >= x_tip - cfg.dcm_r_max)
    )
    upper_idx = np.flatnonzero(upper_mask)
    lower_idx = np.flatnonzero(lower_mask)
    if upper_idx.size == 0 or lower_idx.size == 0:
        return {
            "available": False,
            "reason": "no_crack_face_nodes_in_window",
            "window": [cfg.dcm_r_min, cfg.dcm_r_max],
            "y_band": y_band,
            "crack_len": crack_len,
        }

    upper = np.column_stack([x[upper_idx], uy[upper_idx]])
    lower = np.column_stack([x[lower_idx], uy[lower_idx]])

    bin_edges = np.linspace(x_tip - cfg.dcm_r_max, x_tip - cfg.dcm_r_min, cfg.dcm_n_bins + 1)
    records = []
    material = Material(elastic_modulus=E_scalar, poisson_ratio=nu, plane_strain=not plane_stress)

    for i in range(cfg.dcm_n_bins):
        x0, x1 = float(bin_edges[i]), float(bin_edges[i + 1])
        u_sel = upper[(upper[:, 0] >= x0) & (upper[:, 0] < x1), 1]
        l_sel = lower[(lower[:, 0] >= x0) & (lower[:, 0] < x1), 1]
        if u_sel.size == 0 or l_sel.size == 0:
            continue

        uy_upper = float(np.median(u_sel) if cfg.dcm_use_median else np.mean(u_sel))
        uy_lower = float(np.median(l_sel) if cfg.dcm_use_median else np.mean(l_sel))
        x_mid = 0.5 * (x0 + x1)
        r = float(x_tip - x_mid)
        if r <= 0:
            continue
        records.append(CrackFaceDisplacement(r=r, uy_upper=uy_upper, uy_lower=uy_lower))

    if not records:
        return {
            "available": False,
            "reason": "no_valid_dcm_pairs",
            "window": [cfg.dcm_r_min, cfg.dcm_r_max],
            "y_band": y_band,
            "crack_len": crack_len,
        }

    dcm_stats = estimate_plateau_ki(records, material, use_median=cfg.dcm_use_median)
    ki_vals = np.asarray([r["KI"] for r in dcm_stats.get("samples", [])], dtype=float)
    return {
        "available": True,
        "window": [cfg.dcm_r_min, cfg.dcm_r_max],
        "y_band": y_band,
        "n_pairs": int(dcm_stats["n_samples"]),
        "n_inliers": int(dcm_stats.get("n_inliers", dcm_stats["n_samples"])),
        "fit_model": dcm_stats.get("fit_model", "pointwise_plateau"),
        "fit_r2": dcm_stats.get("fit_r2"),
        "KI_ref": float(dcm_stats["KI_ref"]),
        "KI_mean": float(dcm_stats["KI_mean"]),
        "KI_std": float(dcm_stats["KI_std"]),
        "KI_relative_span": relative_span(ki_vals.tolist(), float(dcm_stats["KI_ref"])),
        "KI_pointwise_median": float(dcm_stats.get("KI_pointwise_median", dcm_stats["KI_ref"])),
        "KI_pointwise_mean": float(dcm_stats.get("KI_pointwise_mean", dcm_stats["KI_mean"])),
        "samples": dcm_stats["samples"],
        "crack_len": crack_len,
    }


def q4_shape(xi: float, eta: float):
    N = np.array([
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta)
    ], dtype=float)

    dN_dxi = np.array([
        [-0.25 * (1 - eta), -0.25 * (1 - xi)],
        [ 0.25 * (1 - eta), -0.25 * (1 + xi)],
        [ 0.25 * (1 + eta),  0.25 * (1 + xi)],
        [-0.25 * (1 + eta),  0.25 * (1 - xi)],
    ], dtype=float)
    return N, dN_dxi


def D_matrix(E: float, nu: float, plane_stress: bool) -> np.ndarray:
    if plane_stress:
        c = E / (1.0 - nu**2)
        return c * np.array([[1.0, nu, 0.0],
                             [nu, 1.0, 0.0],
                             [0.0, 0.0, (1.0 - nu) / 2.0]], dtype=float)
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([[1.0 - nu, nu, 0.0],
                         [nu, 1.0 - nu, 0.0],
                         [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]], dtype=float)


def element_stress_at_parent_point(
    xe: np.ndarray,
    ue: np.ndarray,
    E: float,
    nu: float,
    plane_stress: bool,
    xi: float,
    eta: float
):
    _, dN_dxi = q4_shape(xi, eta)
    J = xe.T @ dN_dxi
    detJ = float(np.linalg.det(J))
    if detJ <= 0:
        raise ValueError("Non-positive detJ")
    invJ = np.linalg.inv(J)
    dN_dx = dN_dxi @ invJ.T

    grad_u = ue.T @ dN_dx
    exx = grad_u[0, 0]
    eyy = grad_u[1, 1]
    gxy = grad_u[0, 1] + grad_u[1, 0]
    eps = np.array([exx, eyy, gxy], dtype=float)
    D = D_matrix(E, nu, plane_stress)
    sig = D @ eps
    sxx, syy, txy = sig
    vm = float(np.sqrt(sxx*sxx - sxx*syy + syy*syy + 3.0*txy*txy))
    return sig, vm




def element_area_q4(xe: np.ndarray) -> float:
    """Approximate physical element area using 2x2 Gauss integration."""
    gps = [(-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)),
           (+1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)),
           (+1.0 / np.sqrt(3.0), +1.0 / np.sqrt(3.0)),
           (-1.0 / np.sqrt(3.0), +1.0 / np.sqrt(3.0))]
    area = 0.0
    for xi, eta in gps:
        _, dN_dxi = q4_shape(xi, eta)
        J = xe.T @ dN_dxi
        area += abs(float(np.linalg.det(J)))
    return float(area)


def build_nodal_smoothed_stress(
    pts: np.ndarray,
    conn: np.ndarray,
    u: np.ndarray,
    E_elem: Optional[np.ndarray],
    E_scalar: float,
    nu: float,
    plane_stress: bool,
):
    """
    Build a continuous nodal stress field by area-weighted averaging of element-corner stresses.

    For each Q4 element, stresses are evaluated at the four parent-space corners and
    accumulated onto the corresponding global nodes. This is a simple nodal recovery
    strategy that removes most element-to-element jumps seen with direct elementwise
    stress sampling.
    """
    nnode = pts.shape[0]
    sig_acc = np.zeros((nnode, 3), dtype=float)
    vm_acc = np.zeros(nnode, dtype=float)
    wt_acc = np.zeros(nnode, dtype=float)

    corners = [(-1.0, -1.0), (+1.0, -1.0), (+1.0, +1.0), (-1.0, +1.0)]

    for e, nodes in enumerate(conn):
        xe = pts[nodes, :]
        ue = np.column_stack([u[2 * nodes], u[2 * nodes + 1]])
        E_here = float(E_elem[e]) if E_elem is not None else float(E_scalar)
        area = max(element_area_q4(xe), 1e-30)
        for a, (xi, eta) in enumerate(corners):
            try:
                sig, vm = element_stress_at_parent_point(xe, ue, E_here, nu, plane_stress, xi, eta)
            except Exception:
                continue
            nid = int(nodes[a])
            sig_acc[nid, :] += area * sig
            vm_acc[nid] += area * vm
            wt_acc[nid] += area

    valid = wt_acc > 0.0
    sig_node = np.full((nnode, 3), np.nan, dtype=float)
    vm_node = np.full(nnode, np.nan, dtype=float)
    sig_node[valid, :] = sig_acc[valid, :] / wt_acc[valid, None]
    vm_node[valid] = vm_acc[valid] / wt_acc[valid]
    return sig_node, vm_node


def interpolate_nodal_stress_in_element(
    sig_node: np.ndarray,
    vm_node: np.ndarray,
    nodes: np.ndarray,
    xi: float,
    eta: float,
):
    """Interpolate recovered nodal stresses/von Mises within a containing element."""
    N, _ = q4_shape(xi, eta)
    sig_local = sig_node[nodes, :]
    vm_local = vm_node[nodes]

    if not np.all(np.isfinite(sig_local)):
        return np.array([np.nan, np.nan, np.nan], dtype=float), float('nan')

    sig = N @ sig_local
    vm = float(N @ np.nan_to_num(vm_local, nan=0.0)) if np.any(np.isfinite(vm_local)) else float('nan')
    return sig, vm


def point_in_quad_newton(xe: np.ndarray, x_target: np.ndarray, tol=1e-10, maxiter=20):
    """
    Find (xi,eta) such that x(xi,eta)=x_target for a convex Q4.
    Returns (inside, xi, eta).
    """
    xi = 0.0
    eta = 0.0
    x_target = np.asarray(x_target, float).reshape(2)

    for _ in range(maxiter):
        N, dN_dxi = q4_shape(xi, eta)
        x = N @ xe
        r = x - x_target
        if np.linalg.norm(r) < tol:
            break
        Jmap = xe.T @ dN_dxi
        try:
            delta = np.linalg.solve(Jmap, r)
        except np.linalg.LinAlgError:
            return False, np.nan, np.nan
        xi -= delta[0]
        eta -= delta[1]
        if np.linalg.norm(delta) < tol:
            break

    inside = (xi >= -1.0001) and (xi <= 1.0001) and (eta >= -1.0001) and (eta <= 1.0001)
    return inside, float(xi), float(eta)




def build_spatial_search_data(pts: np.ndarray, conn: np.ndarray, pad_fraction: float = 1.0e-12):
    """
    Precompute cheap spatial filters for point-location during crack-line sampling.

    Returns
    -------
    dict with element bounding boxes, centroids, and a small padding based on mesh size.
    """
    xe_all = pts[conn]  # (ne, 4, 2)
    xmin = xe_all[:, :, 0].min(axis=1)
    xmax = xe_all[:, :, 0].max(axis=1)
    ymin = xe_all[:, :, 1].min(axis=1)
    ymax = xe_all[:, :, 1].max(axis=1)
    centroids = xe_all.mean(axis=1)

    xspan = float(max(np.max(xmax) - np.min(xmin), 1.0))
    yspan = float(max(np.max(ymax) - np.min(ymin), 1.0))
    pad_x = max(float(np.median(np.maximum(xmax - xmin, 0.0))) * 1.0e-6, pad_fraction * xspan)
    pad_y = max(float(np.median(np.maximum(ymax - ymin, 0.0))) * 1.0e-6, pad_fraction * yspan)

    return {
        "xmin": xmin - pad_x,
        "xmax": xmax + pad_x,
        "ymin": ymin - pad_y,
        "ymax": ymax + pad_y,
        "centroids": centroids,
    }


def candidate_elements_for_point(xq: np.ndarray, search_data: dict) -> np.ndarray:
    """
    Fast bounding-box candidate filter for a query point.
    """
    x = float(xq[0])
    y = float(xq[1])
    mask = (
        (search_data["xmin"] <= x) & (x <= search_data["xmax"]) &
        (search_data["ymin"] <= y) & (y <= search_data["ymax"])
    )
    idx = np.flatnonzero(mask)
    if idx.size <= 1:
        return idx

    c = search_data["centroids"][idx]
    d2 = (c[:, 0] - x) ** 2 + (c[:, 1] - y) ** 2
    order = np.argsort(d2)
    return idx[order]


def sample_stress_line(
    pts: np.ndarray,
    conn: np.ndarray,
    u: np.ndarray,
    tip: np.ndarray,
    crack_dir: np.ndarray,
    r_vals: np.ndarray,
    E_elem: Optional[np.ndarray],
    E_scalar: float,
    nu: float,
    plane_stress: bool,
):
    """
    Sample sigma_yy and von Mises along the crack extension line using a recovered,
    nodally averaged stress field.

    This is intentionally different from the raw elementwise stress evaluation because
    direct gradients of a Q4 displacement field are discontinuous across elements and can
    produce the sign flips/jagged behavior seen in the crack-line plots.
    """
    sigma_yy = np.full_like(r_vals, np.nan, dtype=float)
    von_mises = np.full_like(r_vals, np.nan, dtype=float)

    logging.info("Building nodally averaged stress field for crack-line sampling...")
    sig_node, vm_node = build_nodal_smoothed_stress(
        pts=pts,
        conn=conn,
        u=u,
        E_elem=E_elem,
        E_scalar=E_scalar,
        nu=nu,
        plane_stress=plane_stress,
    )

    search_data = build_spatial_search_data(pts, conn)
    prev_e: Optional[int] = None

    for i, r in enumerate(r_vals):
        xq = tip + r * crack_dir

        candidate_ids = candidate_elements_for_point(xq, search_data)
        if prev_e is not None:
            if candidate_ids.size == 0:
                candidate_ids = np.array([prev_e], dtype=int)
            elif candidate_ids[0] != prev_e:
                candidate_ids = np.concatenate([
                    np.array([prev_e], dtype=int),
                    candidate_ids[candidate_ids != prev_e]
                ])

        found = False
        for e in candidate_ids:
            nodes = conn[int(e)]
            xe = pts[nodes, :]
            inside, xi, eta = point_in_quad_newton(xe, xq)
            if not inside:
                continue

            sig, vm = interpolate_nodal_stress_in_element(sig_node, vm_node, nodes, xi, eta)
            if not np.all(np.isfinite(sig)):
                ue = np.column_stack([u[2 * nodes], u[2 * nodes + 1]])
                E_here = float(E_elem[int(e)]) if E_elem is not None else float(E_scalar)
                sig, vm = element_stress_at_parent_point(xe, ue, E_here, nu, plane_stress, xi, eta)

            sigma_yy[i] = float(sig[1])
            von_mises[i] = float(vm)
            prev_e = int(e)
            found = True
            break

        if not found and prev_e is not None:
            c = search_data["centroids"]
            d2 = (c[:, 0] - xq[0]) ** 2 + (c[:, 1] - xq[1]) ** 2
            nearby = np.argsort(d2)[:16]
            for e in nearby:
                nodes = conn[int(e)]
                xe = pts[nodes, :]
                inside, xi, eta = point_in_quad_newton(xe, xq)
                if not inside:
                    continue
                sig, vm = interpolate_nodal_stress_in_element(sig_node, vm_node, nodes, xi, eta)
                if not np.all(np.isfinite(sig)):
                    ue = np.column_stack([u[2 * nodes], u[2 * nodes + 1]])
                    E_here = float(E_elem[int(e)]) if E_elem is not None else float(E_scalar)
                    sig, vm = element_stress_at_parent_point(xe, ue, E_here, nu, plane_stress, xi, eta)
                sigma_yy[i] = float(sig[1])
                von_mises[i] = float(vm)
                prev_e = int(e)
                found = True
                break

        if (i + 1) % max(1, len(r_vals) // 10) == 0:
            logging.info(
                f"Stress-line sampling progress: {i + 1}/{len(r_vals)} points "
                f"({100.0 * (i + 1) / len(r_vals):.0f}%)"
            )

    return sigma_yy, von_mises

def plot_stress_line(r_vals, sigma_yy, vm, out_png: Path):
    plt.figure()
    plt.plot(r_vals, sigma_yy, label=r"$\sigma_{yy}$")
    plt.plot(r_vals, vm, label="von Mises")
    plt.xscale("log")
    plt.xlabel("r ahead of tip (m)")
    plt.ylabel("Stress (Pa)")
    plt.title("Stress along crack line (theta=0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_sigma_sqrt_2pir(r_vals, sigma_yy, KI_ref, out_png: Path):
    valid = np.isfinite(sigma_yy) & (r_vals > 0.0)
    y = sigma_yy[valid] * np.sqrt(2.0 * np.pi * r_vals[valid])

    plt.figure()
    plt.plot(r_vals[valid], y, label=r"$\sigma_{yy}\sqrt{2\pi r}$")
    plt.axhline(KI_ref, linestyle="--", label=r"$K_I$ (from J)")
    plt.xscale("log")
    plt.xlabel("r ahead of tip (m)")
    plt.ylabel(r"$\sigma_{yy}\sqrt{2\pi r}$ (Pa*sqrt(m))")
    plt.title("LEFM near-tip check along crack line (theta=0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_J_vs_rout(r_outs, J_vals, best_idx, out_png: Path):
    plt.figure()
    plt.plot(r_outs, J_vals, marker="o")
    if best_idx is not None and 0 <= best_idx < len(r_outs):
        plt.plot(r_outs[best_idx], J_vals[best_idx], marker="s", markersize=10)
    plt.xlabel("r_out (m)")
    plt.ylabel("J (N/m)")
    plt.title("J-integral path independence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_KI_vs_rout(r_outs, KI_vals, out_png: Path):
    plt.figure()
    plt.plot(r_outs, KI_vals, marker="o")
    plt.xlabel("r_out (m)")
    plt.ylabel("K_I from J (Pa*sqrt(m))")
    plt.title("K_I vs J-domain outer radius")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
    
def plot_KII_vs_rout(r_outs, KII_vals, out_png: Path):
    plt.figure()
    plt.plot(r_outs, KII_vals, marker="o")
    plt.xlabel("r_out (m)")
    plt.ylabel("K_II from interaction integral (Pa*sqrt(m))")
    plt.title("K_II vs J-domain outer radius")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_dcm_ki_vs_r(samples: List[Dict[str, Any]], ki_ref: float, out_png: Path):
    if not samples:
        return
    r = np.asarray([float(s["r"]) for s in samples], dtype=float)
    ki = np.asarray([float(s["KI"]) for s in samples], dtype=float)
    valid = np.isfinite(r) & np.isfinite(ki) & (r > 0.0)
    if not np.any(valid):
        return

    order = np.argsort(r[valid])
    r_use = r[valid][order]
    ki_use = ki[valid][order]

    plt.figure()
    plt.plot(r_use, ki_use, marker="o", label="DCM pointwise $K_I$")
    plt.axhline(float(ki_ref), linestyle="--", label="DCM fitted $K_I$")
    plt.xscale("log")
    plt.xlabel("r behind tip (m)")
    plt.ylabel("DCM $K_I$ (Pa*sqrt(m))")
    plt.title("DCM $K_I$ vs radial distance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()



def write_csv(path: Path, r_vals, sigma_yy, vm):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["r", "sigma_yy", "von_mises"])
        for r, s, v in zip(r_vals, sigma_yy, vm):
            w.writerow([float(r), float(s) if np.isfinite(s) else "", float(v) if np.isfinite(v) else ""])


def choose_best_idx(r_outs: List[float], KI_vals: List[float]) -> int:
    """
    Simple plateau-ish choice:
    pick the contour with the smallest deviation from the mean of the last half.
    """
    arr = np.asarray(KI_vals, dtype=float)
    n = len(arr)
    tail = arr[n // 2:]
    ref = float(np.mean(tail))
    idx = int(np.argmin(np.abs(arr - ref)))
    return idx


def relative_span(vals: List[float], ref_val: float) -> float:
    arr = np.asarray(vals, dtype=float)
    if abs(ref_val) < 1e-30:
        return float("nan")
    return float((np.max(arr) - np.min(arr)) / abs(ref_val))

def _safe_cov(std: float, mean: float) -> float:
    if abs(mean) < 1e-30:
        return float("nan")
    return float(std / mean)


def _metric_stats(values: List[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "mean": mean,
        "std": std,
        "cov": _safe_cov(std, mean),
        "p5": float(np.percentile(arr, 5.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def run_one_validation(
    cfg: ValConfig,
    realization_id: Optional[int],
    manifest_hash: Optional[str] = None,
) -> Dict[str, Any]:
    suffix = f"_mc{realization_id:04d}" if realization_id is not None else ""
    npz_path = cfg.run_dir / f"fields{suffix}.npz"
    meta_path = cfg.run_dir / f"metadata{suffix}.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    pts, conn, u, E_elem = load_solution(npz_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    tip = crack_tip_xy(meta)
    crack_start = crack_start_xy(meta)
    crack_dir = crack_dir_xy(meta)

    E_scalar = float(np.mean(E_elem)) if E_elem is not None else float(meta.get("E_mean", meta.get("E", cfg.E)))
    nu = float(meta.get("nu", cfg.nu))
    plane_stress = bool(meta.get("plane_stress", cfg.plane_stress))

    use_interaction = cfg.use_interaction_integral_for_stochastic and sweep_interaction_rout is not None
    if cfg.use_interaction_integral_for_stochastic and sweep_interaction_rout is None:
        logging.warning(
            "Interaction integral requested but sweep_interaction_rout is unavailable. Falling back to corrected_J_star."
        )

    if use_interaction:
        E_tip = cfg.E_tip_for_aux if cfg.E_tip_for_aux is not None else E_scalar

        sweep = sweep_interaction_rout(
            pts=pts,
            conn=conn,
            U=u,
            tip=tip,
            E=E_scalar,
            E_elem=E_elem,
            E_tip=E_tip,
            nu=nu,
            plane_stress=plane_stress,
            r_in=cfg.r_in,
            r_out_list=list(cfg.r_out_list),
            crack_dir=crack_dir,
            log_each=True,
            crack_start=crack_start,
            crack_end=tip,
            exclude_crack_faces=True,
            crack_face_exclusion=cfg.crack_face_exclusion,
            modes=cfg.interaction_modes,
            use_inhomogeneity_correction=cfg.interaction_use_inhomogeneity_correction,
        )

        r_outs = [float(s.r_out) for s in sweep]
        KI_signed_vals = [float(s.KI) for s in sweep]
        KI_vals = [abs(v) for v in KI_signed_vals] if cfg.interaction_take_abs_KI else KI_signed_vals
        KII_vals = [float(s.KII) for s in sweep]
        J_vals = None

    else:
        sweep = sweep_J_rout(
            pts=pts,
            conn=conn,
            U=u,
            tip=tip,
            E=E_scalar,
            E_elem=E_elem,
            E_for_KI=E_scalar,
            nu=nu,
            plane_stress=plane_stress,
            r_in=cfg.r_in,
            r_out_list=list(cfg.r_out_list),
            crack_dir=crack_dir,
            log_each=True,
            crack_start=crack_start,
            crack_end=tip,
            exclude_crack_faces=True,
            crack_face_exclusion=cfg.crack_face_exclusion,
        )

        r_outs = [float(s.r_out) for s in sweep]
        J_vals = [float(s.J) for s in sweep]
        KI_vals = [float(s.KI) for s in sweep]
        KII_vals = None

    best_idx = choose_best_idx(r_outs, KI_vals)

    KI_ref = KI_vals[best_idx]
    KI_relative_span = relative_span(KI_vals, KI_ref)

    if J_vals is not None:
        J_ref = J_vals[best_idx]
        J_relative_span = relative_span(J_vals, J_ref)
        JK_relative_residual = abs(
            KI_ref - np.sqrt(
                max(J_ref, 0.0) *
                (E_scalar if plane_stress else E_scalar / (1.0 - nu**2))
            )
        )
        JK_relative_residual /= max(abs(KI_ref), 1e-30)
    else:
        J_ref = None
        J_relative_span = None
        JK_relative_residual = None

    r_vals = np.logspace(np.log10(cfg.r_min), np.log10(cfg.r_max), cfg.n_r)
    sigma_yy, vm = sample_stress_line(
        pts=pts,
        conn=conn,
        u=u,
        tip=tip,
        crack_dir=crack_dir,
        r_vals=r_vals,
        E_elem=E_elem,
        E_scalar=E_scalar,
        nu=nu,
        plane_stress=plane_stress,
    )

    if J_vals is not None:
        plot_J_vs_rout(r_outs, J_vals, best_idx, cfg.run_dir / f"J_path_independence{suffix}.png")
        
    if KII_vals is not None:
        plot_KII_vs_rout(r_outs, KII_vals, cfg.run_dir / f"KII_vs_rout{suffix}.png")

    plot_KI_vs_rout(r_outs, KI_vals, cfg.run_dir / f"KI_vs_rout{suffix}.png")
    plot_stress_line(r_vals, sigma_yy, vm, cfg.run_dir / f"sigma_yy_and_vonmises_vs_r{suffix}.png")
    plot_sigma_sqrt_2pir(r_vals, sigma_yy, KI_ref, cfg.run_dir / f"sigma_sqrt_2pir_vs_r{suffix}.png")

    if cfg.export_csv:
        write_csv(cfg.run_dir / f"validation_line{suffix}.csv", r_vals, sigma_yy, vm)

    summary = {
    "run_name": cfg.run_dir.name,
    "realization_id": realization_id,
    "seed": meta.get("random_seed"),
    "nsamples": meta.get("n_realizations"),
    "units": {
        "r": "m",
        "J_star": "J/m^2",
        "KI": "Pa*sqrt(m)",
        "KII": "Pa*sqrt(m)",
        "E": "Pa",
    },
    "geometry_type": meta.get("geometry_type"),
    "a": meta.get("a"),
    "tip": tip.tolist(),
    "crack_start": crack_start.tolist(),
    "crack_dir": crack_dir.tolist(),
    "E_mean_scalar_used_in_post": E_scalar,
    "method": "interaction_integral" if use_interaction else "corrected_J_star",
    "interaction_settings": {
        "requested": bool(cfg.use_interaction_integral_for_stochastic),
        "applied": bool(use_interaction),
        "modes": list(cfg.interaction_modes),
        "use_inhomogeneity_correction": bool(cfg.interaction_use_inhomogeneity_correction),
        "take_abs_KI": bool(cfg.interaction_take_abs_KI),
        "E_tip_for_aux": float(E_tip) if use_interaction else (float(cfg.E_tip_for_aux) if cfg.E_tip_for_aux is not None else None),
        "interaction_impl_available": bool(sweep_interaction_rout is not None),
    },
    "r_in": cfg.r_in,
    "r_out_list": r_outs,
    "KI_list": KI_vals,
    "best_idx": best_idx,
    "best_r_out": r_outs[best_idx],
    "KI_ref": KI_ref,
    "KI_relative_span": KI_relative_span,
    "E_elem_mean": float(np.mean(E_elem)) if E_elem is not None else E_scalar,
    "E_elem_std": float(np.std(E_elem)) if E_elem is not None else 0.0,
    "manifest_hash_sha256": manifest_hash,
    }

    if J_vals is not None:
        summary["J_list"] = J_vals
        summary["J_ref"] = J_ref
        summary["J_relative_span"] = J_relative_span
        summary["JK_relative_residual"] = float(JK_relative_residual)

    if KII_vals is not None:
        summary["KII_list"] = KII_vals
        summary["KII_ref"] = float(KII_vals[best_idx])
    if use_interaction:
        summary["KI_signed_list"] = KI_signed_vals
        summary["KI_ref_signed"] = float(KI_signed_vals[best_idx])

    if cfg.enable_dcm_from_fields:
        dcm_summary = estimate_dcm_from_fields(
            pts=pts,
            conn=conn,
            u=u,
            tip=tip,
            crack_start=crack_start,
            cfg=cfg,
            E_scalar=E_scalar,
            nu=nu,
            plane_stress=plane_stress,
        )
        summary["dcm"] = dcm_summary
        if dcm_summary.get("available"):
            ki_dcm = float(dcm_summary["KI_ref"])
            summary["KI_ref_dcm"] = ki_dcm
            summary["KI_ref_dcm_delta_to_J"] = float((ki_dcm - KI_ref) / max(abs(KI_ref), 1e-30))
            plot_dcm_ki_vs_r(
                samples=dcm_summary.get("samples", []),
                ki_ref=ki_dcm,
                out_png=cfg.run_dir / f"dcm_KI_vs_r{suffix}.png",
            )
        
    (cfg.run_dir / f"validation_summary{suffix}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    logging.info(f"Finished validation for realization {realization_id}.")
    return summary


def write_all_realization_summaries(run_dir: Path, summaries: List[dict]) -> None:
    if not summaries:
        return

    ki_refs = [float(s["KI_ref"]) for s in summaries if s.get("KI_ref") is not None]
    j_refs = [float(s["J_ref"]) for s in summaries if s.get("J_ref") is not None]

    aggregate = {
        "metadata": {
            "run_name": summaries[0].get("run_name", run_dir.name),
            "seed": summaries[0].get("seed"),
            "nsamples": len(summaries),
            "units": summaries[0].get("units", {}),
        },
        "metrics_by_realization_set": {
            "KI_ref": _metric_stats(ki_refs) if ki_refs else None,
            "J_star_ref": _metric_stats(j_refs) if j_refs else None,
        },
        "realizations": summaries,
    }
    (run_dir / "validation_summary_all_realizations.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    csv_path = run_dir / "validation_summary_all_realizations.csv"
    fieldnames = [
        "run_name",
        "seed",
        "nsamples",
        "realization_id",
        "best_r_out",
        "KI_ref",
        "J_star_ref",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(
                {
                    "run_name": s.get("run_name"),
                    "seed": s.get("seed"),
                    "nsamples": s.get("nsamples"),
                    "realization_id": s.get("realization_id"),
                    "best_r_out": s.get("best_r_out"),
                    "KI_ref": s.get("KI_ref"),
                    "J_star_ref": s.get("J_ref"),
                }
            )
def write_aggregate_summary(cfg: ValConfig, summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        return
    ki = np.asarray([float(s["KI_ref"]) for s in summaries], dtype=float)
    rec: Dict[str, Any] = {
        "run_name": cfg.run_dir.name,
        "n_realizations": int(len(summaries)),
        "realization_ids": [s.get("realization_id") for s in summaries],
        "KI_ref_mean": float(np.mean(ki)),
        "KI_ref_std": float(np.std(ki)),
        "KI_ref_cov": float(np.std(ki) / max(abs(np.mean(ki)), 1e-30)),
        "KI_ref_min": float(np.min(ki)),
        "KI_ref_max": float(np.max(ki)),
        "summaries": summaries,
    }

    dcm_vals = [s.get("KI_ref_dcm") for s in summaries if s.get("KI_ref_dcm") is not None]
    if dcm_vals:
        d = np.asarray(dcm_vals, dtype=float)
        rec["KI_ref_dcm_mean"] = float(np.mean(d))
        rec["KI_ref_dcm_std"] = float(np.std(d))
        rec["KI_ref_dcm_cov"] = float(np.std(d) / max(abs(np.mean(d)), 1e-30))

    (cfg.run_dir / cfg.aggregate_summary_name).write_text(json.dumps(rec, indent=2), encoding="utf-8")


def _parse_crack_mm(name: str) -> Optional[float]:
    import re
    m = re.search(r"(\d+(?:\.\d+)?)mm", name)
    return float(m.group(1)) if m else None


def _collect_ki_vs_a_from_summaries(root: Path, glob_pat: str, stochastic: bool) -> List[Dict[str, float]]:
    rows = []
    for run_dir in sorted(root.glob(glob_pat)):
        a_mm = _parse_crack_mm(run_dir.name)
        if a_mm is None:
            continue
        if stochastic:
            p = run_dir / "validation_summary_all_realizations.json"
            if not p.exists():
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            ki = float(data["KI_ref_mean"])
        else:
            p = run_dir / "validation_summary.json"
            if not p.exists():
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            ki = float(data["KI_ref"])
        rows.append({"a": a_mm * 1e-3, "KI": ki, "run_dir": str(run_dir)})
    rows.sort(key=lambda r: r["a"])
    return rows


def _write_ki_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["a", "KI", "run_dir"])
        w.writeheader()
        w.writerows(rows)


def _run_lifing_chain(cfg: ValConfig, ki_csv: Path, out_dir: Path, tag: str) -> None:
    from Lifing.fatigue_lifing_utils import compute_delta_k_from_R, integrate_crack_growth
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.genfromtxt(ki_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    a = np.asarray(arr["a"], dtype=float)
    ki = np.asarray(arr["KI"], dtype=float)
    dk = compute_delta_k_from_R(ki, cfg.life_R)
    delta_csv = out_dir / f"delta_k_curve_{tag}.csv"
    with open(delta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["a", "DeltaK"])
        for ai, dki in zip(a, dk):
            w.writerow([float(ai), float(dki)])

    N, dadn = integrate_crack_growth(a, dk / 1e6, cfg.paris_C, cfg.paris_m)
    life_det = float(N[-1]) if len(N) else float("nan")

    rng = np.random.default_rng(42)
    C_samples = rng.normal(cfg.paris_C, cfg.paris_C * cfg.life_C_cov, cfg.life_nsamples)
    m_samples = rng.normal(cfg.paris_m, cfg.life_m_std, cfg.life_nsamples)
    sigma_scales = rng.normal(cfg.life_sigma_scale_mean, cfg.life_sigma_scale_mean * cfg.life_sigma_scale_cov, cfg.life_nsamples)
    life_samples = []
    for C, m, s in zip(C_samples, m_samples, sigma_scales):
        N_i, _ = integrate_crack_growth(a, (dk * s) / 1e6, float(C), float(m))
        life_samples.append(float(N_i[-1]))
    life_samples = np.asarray(life_samples, dtype=float)

    plt.figure()
    plt.plot(a, dk)
    plt.xlabel("a (m)")
    plt.ylabel("DeltaK (Pa*sqrt(m))")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"delta_k_{tag}.png", dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(np.maximum(N, 1.0), a)
    plt.xscale("log")
    plt.xlabel("Cycles N")
    plt.ylabel("a (m)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"a_vs_N_{tag}.png", dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(np.log10(np.maximum(life_samples, 1.0)), bins=40)
    plt.xlabel("log10(cycles)")
    plt.ylabel("count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"life_histogram_{tag}.png", dpi=220, bbox_inches="tight")
    plt.close()

    stats = {
        "tag": tag,
        "n_points": int(len(a)),
        "life_det": life_det,
        "life_mc_mean": float(np.mean(life_samples)),
        "life_mc_std": float(np.std(life_samples)),
        "life_mc_p05": float(np.percentile(life_samples, 5)),
        "life_mc_p50": float(np.percentile(life_samples, 50)),
        "life_mc_p95": float(np.percentile(life_samples, 95)),
    }
    (out_dir / f"lifing_stats_{tag}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def run_crack_length_lifing_comparison(cfg: ValConfig) -> None:
    out_dir = cfg.lifing_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    det_rows = _collect_ki_vs_a_from_summaries(cfg.sweep_root, cfg.deterministic_glob, stochastic=False)
    stoch_rows = _collect_ki_vs_a_from_summaries(cfg.sweep_root, cfg.stochastic_glob, stochastic=True)
    if not det_rows or not stoch_rows:
        logging.warning("Skipping crack-length lifing comparison. Missing deterministic or stochastic validated runs.")
        return

    n = min(len(det_rows), len(stoch_rows))
    det_rows = det_rows[:n]
    stoch_rows = stoch_rows[:n]

    det_csv = out_dir / "ki_vs_a_deterministic.csv"
    stoch_csv = out_dir / "ki_vs_a_stochastic.csv"
    _write_ki_csv(det_csv, det_rows)
    _write_ki_csv(stoch_csv, stoch_rows)
    _run_lifing_chain(cfg, det_csv, out_dir, "deterministic")
    _run_lifing_chain(cfg, stoch_csv, out_dir, "stochastic")

def main():
    setup_logging()
    cfg = ValConfig()
    runtime_cfg_path = cfg.run_dir / "runtime_config.json"
    runtime_cfg = load_runtime_config(runtime_cfg_path)
    mat_cfg = runtime_cfg.get("material", {})
    if mat_cfg:
        cfg.E = float(mat_cfg.get("E", mat_cfg.get("E_mean", cfg.E)))
        cfg.nu = float(mat_cfg.get("nu", cfg.nu))
        cfg.plane_stress = bool(mat_cfg.get("plane_stress", cfg.plane_stress))
    val_cfg = runtime_cfg.get("validation", {})
    stage_val_cfg = runtime_cfg.get("stages", {}).get("validation", {})
    merged_val_cfg = {
        **(val_cfg if isinstance(val_cfg, dict) else {}),
        **(stage_val_cfg if isinstance(stage_val_cfg, dict) else {}),
    }
    if merged_val_cfg:
        cfg.run_all_realizations = _as_bool(
            merged_val_cfg.get("run_all_realizations"),
            cfg.run_all_realizations,
        )
        cfg.realization_glob = str(merged_val_cfg.get("realization_glob", cfg.realization_glob))
        if merged_val_cfg.get("realization_id") is not None:
            cfg.realization_id = int(merged_val_cfg["realization_id"])
        cfg.use_interaction_integral_for_stochastic = _as_bool(
            merged_val_cfg.get(
                "use_interaction_integral_for_stochastic",
                merged_val_cfg.get("use_interaction_integral"),
            ),
            cfg.use_interaction_integral_for_stochastic,
        )
        if merged_val_cfg.get("E_tip_for_aux") is not None:
            cfg.E_tip_for_aux = float(merged_val_cfg.get("E_tip_for_aux"))
        cfg.interaction_modes = _as_interaction_modes(
            merged_val_cfg.get("interaction_modes"),
            cfg.interaction_modes,
        )
        cfg.interaction_use_inhomogeneity_correction = _as_bool(
            merged_val_cfg.get("interaction_use_inhomogeneity_correction"),
            cfg.interaction_use_inhomogeneity_correction,
        )
        cfg.interaction_take_abs_KI = _as_bool(
            merged_val_cfg.get("interaction_take_abs_KI"),
            cfg.interaction_take_abs_KI,
        )

        # Backward compatible aliases for DCM toggles.
        dcm_enabled_raw = merged_val_cfg.get(
            "enable_dcm_from_fields",
            merged_val_cfg.get("enable_dcm", merged_val_cfg.get("dcm")),
        )
        cfg.enable_dcm_from_fields = _as_bool(dcm_enabled_raw, cfg.enable_dcm_from_fields)
        cfg.dcm_r_min = float(merged_val_cfg.get("dcm_r_min", cfg.dcm_r_min))
        cfg.dcm_r_max = float(merged_val_cfg.get("dcm_r_max", cfg.dcm_r_max))
        cfg.dcm_n_bins = int(merged_val_cfg.get("dcm_n_bins", cfg.dcm_n_bins))
        cfg.dcm_y_band_scale = float(merged_val_cfg.get("dcm_y_band_scale", cfg.dcm_y_band_scale))
        cfg.dcm_use_median = _as_bool(merged_val_cfg.get("dcm_use_median"), cfg.dcm_use_median)
    manifest_path = cfg.run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = load_run_manifest(cfg.run_dir)
        manifest_hash = manifest.get("manifest_hash_sha256")
    else:
        _, manifest_hash = write_run_manifest(
            cfg.run_dir,
            {
                "workflow": "Stochastic_FEM.stochastic_validate_fields",
                "geometry_mesh": {},
                "solver": {
                    "E": cfg.E,
                    "nu": cfg.nu,
                    "plane_stress": cfg.plane_stress,
                    "sigma_nominal": cfg.sigma_nominal,
                },
                "validation": {
                    "r_min": cfg.r_min,
                    "r_max": cfg.r_max,
                    "n_r": cfg.n_r,
                    "r_in": cfg.r_in,
                    "r_out_list": list(cfg.r_out_list),
                    "crack_face_exclusion": cfg.crack_face_exclusion,
                    "use_interaction_integral_for_stochastic": cfg.use_interaction_integral_for_stochastic,
                    "interaction_modes": list(cfg.interaction_modes),
                    "interaction_use_inhomogeneity_correction": cfg.interaction_use_inhomogeneity_correction,
                    "interaction_take_abs_KI": cfg.interaction_take_abs_KI,
                    "E_tip_for_aux": cfg.E_tip_for_aux,
                },
                "rng": {"seed_derivation_rule": "realization_seed = base_seed + realization_id"},
            },
        )

    if cfg.run_all_realizations:
        ids = discover_realization_ids(cfg.run_dir, cfg.realization_glob)
        if not ids:
            raise FileNotFoundError(
                f"No realization files found in {cfg.run_dir} matching {cfg.realization_glob}"
            )

        logging.info(f"Found realization IDs: {ids}")
        summaries: List[Dict[str, Any]] = []
        for rid in ids:
            summaries.append(run_one_validation(cfg, rid, manifest_hash=manifest_hash))
        write_all_realization_summaries(cfg.run_dir, summaries)
        write_aggregate_summary(cfg, summaries)
        update_runtime_config(
            runtime_cfg_path,
            stage="validation",
            updates={"stage": {"validated_realization_ids": ids, "summary_pattern": "validation_summary_mc*.json"}},
        )
    else:
        run_one_validation(cfg, cfg.realization_id, manifest_hash=manifest_hash)
        update_runtime_config(
            runtime_cfg_path,
            stage="validation",
            updates={"stage": {"validated_realization_ids": [cfg.realization_id], "summary_pattern": "validation_summary*.json"}},
        )

    if cfg.run_crack_length_sweep:
        run_crack_length_lifing_comparison(cfg)


if __name__ == "__main__":
    main()

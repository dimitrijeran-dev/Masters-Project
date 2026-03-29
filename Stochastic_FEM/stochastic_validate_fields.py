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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List

from src.configs.run_io import load_runtime_config, update_runtime_config

import matplotlib.pyplot as plt
import meshio
import numpy as np

import sys

THIS = Path(__file__).resolve()
ROOT = THIS.parent
sys.path.append(str(ROOT))

# Prefer the stochastic-capable J-integral if present in the same folder.
try:
    from J_Integral_stochastic_modified import (
        sweep_J_rout,
        sweep_interaction_rout,
    )  # type: ignore
except Exception:
    from J_Integral import sweep_J_rout  # type: ignore
    sweep_interaction_rout = None


@dataclass
class ValConfig:
    run_dir: Path = Path("Data/New Data/Deterministic_config_plate_edge_crack")
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

    # Output control
    export_csv: bool = True
    csv_name: str = "validation_line.csv"
    summary_name: str = "validation_summary.json"


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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

def run_one_validation(cfg: ValConfig, realization_id: Optional[int]) -> None:
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

    use_interaction = (
    cfg.use_interaction_integral_for_stochastic
    and E_elem is not None
    and sweep_interaction_rout is not None
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
            modes=("I", "II"),
        )

        r_outs = [float(s.r_out) for s in sweep]
        KI_vals = [float(s.KI) for s in sweep]
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
    "geometry_type": meta.get("geometry_type"),
    "a": meta.get("a"),
    "tip": tip.tolist(),
    "crack_start": crack_start.tolist(),
    "crack_dir": crack_dir.tolist(),
    "E_mean_scalar_used_in_post": E_scalar,
    "method": "interaction_integral" if use_interaction else "corrected_J_star",
    "r_in": cfg.r_in,
    "r_out_list": r_outs,
    "KI_list": KI_vals,
    "best_idx": best_idx,
    "best_r_out": r_outs[best_idx],
    "KI_ref": KI_ref,
    "KI_relative_span": KI_relative_span,
    "E_elem_mean": float(np.mean(E_elem)) if E_elem is not None else E_scalar,
    "E_elem_std": float(np.std(E_elem)) if E_elem is not None else 0.0,
    }

    if J_vals is not None:
        summary["J_list"] = J_vals
        summary["J_ref"] = J_ref
        summary["J_relative_span"] = J_relative_span
        summary["JK_relative_residual"] = float(JK_relative_residual)

    if KII_vals is not None:
        summary["KII_list"] = KII_vals
        summary["KII_ref"] = float(KII_vals[best_idx])
        
    (cfg.run_dir / f"validation_summary{suffix}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    logging.info(f"Finished validation for realization {realization_id}.")

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

    if cfg.run_all_realizations:
        ids = discover_realization_ids(cfg.run_dir, cfg.realization_glob)
        if not ids:
            raise FileNotFoundError(
                f"No realization files found in {cfg.run_dir} matching {cfg.realization_glob}"
            )

        logging.info(f"Found realization IDs: {ids}")
        for rid in ids:
            run_one_validation(cfg, rid)
        update_runtime_config(
            runtime_cfg_path,
            stage="validation",
            updates={"stage": {"validated_realization_ids": ids, "summary_pattern": "validation_summary_mc*.json"}},
        )
    else:
        run_one_validation(cfg, cfg.realization_id)
        update_runtime_config(
            runtime_cfg_path,
            stage="validation",
            updates={"stage": {"validated_realization_ids": [cfg.realization_id], "summary_pattern": "validation_summary*.json"}},
        )


if __name__ == "__main__":
    main()


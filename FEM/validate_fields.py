#!/usr/bin/env python3
"""
validate_fields.py

Validation plots for fracture FE results:
- KI vs r_out from J-integral sweep
- sigma_yy(r) ahead of crack tip and LEFM check: sigma_yy*sqrt(2*pi*r) ~ KI
- von Mises along the same line
- Geometry factor Y = KI / (sigma*sqrt(pi*a)) and (optional) compare across a/W

Inputs:
- Either: a VTU/VTK with point_data U (your solution output)
- Or: a saved npz containing pts, conn, u (recommended for reproducibility)

This script assumes:
- Q4 elements
- plane stress/strain
- your J-integral code returns KI from J via KI = sqrt(J*E')
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import meshio

from datetime import datetime
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.run_manifest import load_run_manifest, write_run_manifest, ensure_summary_manifest_hash

# --- import your J-integral functions ---
from src.J_Integral import compute_J_domain_q4, sweep_J_rout

# ----------------------------
# Config
# ----------------------------
@dataclass
class ValConfig:
    run_name: Optional[str] = None
    run_dir: Optional[Path] = None
    vtu_path: Optional[Path] = None
    npz_path: Optional[Path] = None
    export_csv: Optional[Path] = None

    # Geometry
    a: float = 0.055
    W: float = 0.200
    H: float = 0.100

    # Material
    E: float = 73.1e9
    nu: float = 0.33
    plane_stress: bool = True

    # Loading
    sigma_nominal: float = 50e6

    # Crack definition
    tip: Tuple[float, float] = (0.055, 0.0)
    crack_dir: Tuple[float, float] = (1.0, 0.0)

    # Stress sampling
    r_min: float = 2.0e-4
    r_max: float = 3.0e-2
    n_r: int = 200

    # J sweep
    r_in: float = 0.008
    r_out_list: Tuple[float, ...] = (
        0.014, 0.016, 0.018, 0.020,
        0.022, 0.024, 0.026
    )
    crack_face_exclusion: float = 5.0e-4


# ----------------------------
# Logging
# ----------------------------
def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ----------------------------
# Basic constitutive
# ----------------------------
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


def von_mises_plane_stress(sxx: float, syy: float, txy: float) -> float:
    return float(np.sqrt(sxx*sxx - sxx*syy + syy*syy + 3.0*txy*txy))


def von_mises_plane_strain(E: float, nu: float, exx: float, eyy: float, sxx: float, syy: float, txy: float) -> float:
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    szz = lam * (exx + eyy)  # εzz=0
    return float(np.sqrt(0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2) + 3.0*(txy**2)))

def E_prime(E: float, nu: float, plane_stress: bool) -> float:
    return float(E if plane_stress else E / (1.0 - nu**2))


# ----------------------------
# Q4 shape functions / mapping
# ----------------------------
def q4_shape(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    N = np.array([
        0.25*(1-xi)*(1-eta),
        0.25*(1+xi)*(1-eta),
        0.25*(1+xi)*(1+eta),
        0.25*(1-xi)*(1+eta)
    ], dtype=float)

    dN_dxi = np.array([
        [-0.25*(1-eta), -0.25*(1-xi)],
        [ 0.25*(1-eta), -0.25*(1+xi)],
        [ 0.25*(1+eta),  0.25*(1+xi)],
        [-0.25*(1+eta),  0.25*(1-xi)],
    ], dtype=float)
    return N, dN_dxi


def q4_map_x(xe: np.ndarray, xi: float, eta: float) -> np.ndarray:
    N, _ = q4_shape(xi, eta)
    return N @ xe


def q4_jacobian(xe: np.ndarray, xi: float, eta: float) -> Tuple[np.ndarray, float]:
    _, dN_dxi = q4_shape(xi, eta)
    J = xe.T @ dN_dxi
    detJ = float(np.linalg.det(J))
    return J, detJ


def q4_grad_u_and_strain_stress(
    xe: np.ndarray,
    ue: np.ndarray,
    xi: float,
    eta: float,
    D: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      grad_u (2,2), eps_voigt (3,), sig_voigt (3,)
    """
    _, dN_dxi = q4_shape(xi, eta)
    J = xe.T @ dN_dxi
    detJ = float(np.linalg.det(J))
    if detJ <= 0:
        raise ValueError("Inverted element encountered.")
    invJ = np.linalg.inv(J)
    dN_dx = dN_dxi @ invJ.T  # (4,2)

    grad_u = ue.T @ dN_dx  # (2,2)

    exx = grad_u[0, 0]
    eyy = grad_u[1, 1]
    gxy = grad_u[0, 1] + grad_u[1, 0]
    eps = np.array([exx, eyy, gxy], dtype=float)

    sig = D @ eps
    return grad_u, eps, sig


# ----------------------------
# Data IO
# ----------------------------
def load_solution(cfg: ValConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      pts (N,2), conn (Ne,4), u (2N,)
    """
    if cfg.npz_path is not None:
        data = np.load(cfg.npz_path)
        pts = np.asarray(data["pts"], float)
        conn = np.asarray(data["conn"], int)
        u = np.asarray(data["u"], float).reshape(-1)
        return pts, conn, u

    if cfg.vtu_path is None:
        raise ValueError("Provide either npz_path or vtu_path.")

    m = meshio.read(str(cfg.vtu_path))
    pts = np.asarray(m.points[:, :2], float)

    # connectivity
    quad_blocks = []
    for c in m.cells:
        if c.type in ("quad", "quad4", "quadrilateral"):
            quad_blocks.append(np.asarray(c.data, int))
    if not quad_blocks:
        raise ValueError("No quad cells found in VTU/VTK.")
    conn = np.vstack(quad_blocks)

    if "U" not in m.point_data:
        raise ValueError("Point data 'U' not found in file. Ensure solver wrote point_data={'U':U3}.")
    U3 = np.asarray(m.point_data["U"], float)
    u = np.empty(2 * pts.shape[0], float)
    u[0::2] = U3[:, 0]
    u[1::2] = U3[:, 1]
    return pts, conn, u


# ----------------------------
# Stress sampling along ray from crack tip
# ----------------------------
def element_centroids(pts: np.ndarray, conn: np.ndarray) -> np.ndarray:
    xe = pts[conn]  # (ne,4,2)
    return xe.mean(axis=1)


def find_nearest_elements_to_points(
    centroids: np.ndarray,
    query_pts: np.ndarray,
    k: int = 8
) -> np.ndarray:
    """
    For each query point, return indices of k nearest element centroids.
    Brute-force (OK for a few hundred queries).
    """
    out = np.empty((query_pts.shape[0], k), dtype=int)
    for i, q in enumerate(query_pts):
        d2 = np.sum((centroids - q)**2, axis=1)
        out[i] = np.argsort(d2)[:k]
    return out


def is_point_in_convex_quad(x: np.ndarray, xe: np.ndarray) -> bool:
    """
    Simple convex quad test using consistent orientation of edges.
    Works best for convex quads (which your mesh should mostly be).
    """
    # Treat quad as polygon in order; test using cross products
    p = xe
    sign = None
    for i in range(4):
        a = p[i]
        b = p[(i+1) % 4]
        edge = b - a
        rel = x - a
        cross = edge[0]*rel[1] - edge[1]*rel[0]
        if abs(cross) < 1e-14:
            continue
        s = cross > 0
        if sign is None:
            sign = s
        elif sign != s:
            return False
    return True


def invert_isoparametric_q4(x: np.ndarray, xe: np.ndarray, iters: int = 20) -> Optional[Tuple[float, float]]:
    """
    Newton solve for (xi,eta) such that x = sum N(xi,eta)*xe.
    Returns None if fails.
    """
    xi, eta = 0.0, 0.0
    for _ in range(iters):
        N, dN_dxi = q4_shape(xi, eta)
        xhat = N @ xe
        r = xhat - x
        if np.linalg.norm(r) < 1e-12:
            return float(xi), float(eta)
        J = xe.T @ dN_dxi
        detJ = float(np.linalg.det(J))
        if abs(detJ) < 1e-16:
            return None
        dxi = np.linalg.solve(J, r)  # J * d = r
        # Newton step is negative
        xi -= dxi[0]
        eta -= dxi[1]
        if abs(xi) > 2.0 or abs(eta) > 2.0:
            # drifted away; likely not in element
            return None
    # accept if close to [-1,1]
    if abs(xi) <= 1.05 and abs(eta) <= 1.05:
        return float(xi), float(eta)
    return None


def sample_stress_along_crack_line(
    pts: np.ndarray,
    conn: np.ndarray,
    u: np.ndarray,
    tip: np.ndarray,
    crack_dir: np.ndarray,
    r_vals: np.ndarray,
    E: float,
    nu: float,
    plane_stress: bool,
) -> Dict[str, np.ndarray]:
    """
    Sample sigma_yy and von Mises along ray x = tip + r*crack_dir.

    Returns dict with arrays:
      r, x, y, sxx, syy, txy, von_mises
    """
    D = D_matrix(E, nu, plane_stress)
    cent = element_centroids(pts, conn)

    # query points
    crack_dir = crack_dir / np.linalg.norm(crack_dir)
    Q = tip[None, :] + r_vals[:, None] * crack_dir[None, :]

    # find candidate elements by nearest centroids
    candidates = find_nearest_elements_to_points(cent, Q, k=10)

    sxx = np.full(r_vals.shape, np.nan)
    syy = np.full(r_vals.shape, np.nan)
    txy = np.full(r_vals.shape, np.nan)
    vm  = np.full(r_vals.shape, np.nan)

    for i, xq in enumerate(Q):
        found = False
        for e in candidates[i]:
            nodes = conn[e]
            xe = pts[nodes]
            if not is_point_in_convex_quad(xq, xe):
                continue
            xi_eta = invert_isoparametric_q4(xq, xe)
            if xi_eta is None:
                continue
            xi, eta = xi_eta
            ue = np.column_stack([u[2*nodes], u[2*nodes + 1]])
            grad_u, eps, sig = q4_grad_u_and_strain_stress(xe, ue, xi, eta, D)
            sxx[i], syy[i], txy[i] = sig[0], sig[1], sig[2]
            if plane_stress:
                vm[i] = von_mises_plane_stress(sxx[i], syy[i], txy[i])
            else:
                vm[i] = von_mises_plane_strain(E, nu, eps[0], eps[1], sxx[i], syy[i], txy[i])
            found = True
            break
        if not found:
            # leave as nan; outside mesh or inversion failed
            pass

    return {
        "r": r_vals,
        "x": Q[:, 0],
        "y": Q[:, 1],
        "sxx": sxx,
        "syy": syy,
        "txy": txy,
        "von_mises": vm,
    }

def choose_best_plateau_index(values: List[float]) -> int:
    arr = np.asarray(values, dtype=float)
    n = len(arr)

    if n < 3:
        return n // 2

    best_i = 1
    best_score = np.inf

    for i in range(1, n - 1):
        window = arr[i - 1:i + 2]
        meanv = float(np.mean(window))
        if abs(meanv) < 1e-30:
            continue
        rel_span = (np.max(window) - np.min(window)) / abs(meanv)
        if rel_span < best_score:
            best_score = rel_span
            best_i = i

    return best_i

def relative_span(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    m = float(np.mean(arr))
    if abs(m) < 1e-30:
        return np.nan
    return float((np.max(arr) - np.min(arr)) / abs(m))

def JK_consistency_residual(J: float, KI: float, E: float, nu: float, plane_stress: bool) -> float:
    Ep = E_prime(E, nu, plane_stress)
    J_from_K = KI**2 / Ep
    denom = max(abs(J), 1e-30)
    return float(abs(J_from_K - J) / denom)

# ----------------------------
# Plot helpers
# ----------------------------
def plot_KI_vs_rout(r_out_list: List[float], KI_list: List[float], out_png: Optional[Path] = None):
    plt.figure()
    plt.plot(r_out_list, KI_list, marker="o")
    plt.xlabel("r_out (m)")
    plt.ylabel("K_I from J (Pa*sqrt(m))")
    plt.title("K_I vs J-domain outer radius")
    plt.grid(True)
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")


def plot_sigma_sqrt_r(r: np.ndarray, syy: np.ndarray, KI_ref: float, out_png: Optional[Path] = None):
    mask = np.isfinite(syy) & (r > 0)
    rr = r[mask]
    val = syy[mask] * np.sqrt(2.0*np.pi*rr)

    plt.figure()
    plt.plot(rr, val, label=r"$\sigma_{yy}\sqrt{2\pi r}$")
    plt.axhline(KI_ref, linestyle="--", label=r"$K_I$ (from J)")
    plt.xlabel("r ahead of tip (m)")
    plt.ylabel(r"$\sigma_{yy}\sqrt{2\pi r}$ (Pa*sqrt(m))")
    plt.title("LEFM near-tip check along crack line (theta=0)")
    plt.grid(True)
    plt.legend()
    plt.xscale("log")
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")


def plot_sigma_and_vm(r: np.ndarray, syy: np.ndarray, vm: np.ndarray, out_png: Optional[Path] = None):
    plt.figure()
    mask1 = np.isfinite(syy)
    mask2 = np.isfinite(vm)
    plt.plot(r[mask1], syy[mask1], label=r"$\sigma_{yy}$")
    plt.plot(r[mask2], vm[mask2], label="von Mises")
    plt.xlabel("r ahead of tip (m)")
    plt.ylabel("Stress (Pa)")
    plt.title("Stress along crack line (theta=0)")
    plt.grid(True)
    plt.legend()
    plt.xscale("log")
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        
def plot_J_vs_rout(
    r_out_list: List[float],
    J_list: List[float],
    chosen_idx: Optional[int] = None,
    out_png: Optional[Path] = None,
):
    plt.figure()
    plt.plot(r_out_list, J_list, marker="o")
    if chosen_idx is not None:
        plt.plot(r_out_list[chosen_idx], J_list[chosen_idx], marker="s", markersize=8)
    plt.xlabel("r_out (m)")
    plt.ylabel("J (N/m)")
    plt.title("J-integral path independence")
    plt.grid(True)
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")


def print_Y_factor(KI: float, sigma: float, a: float):
    Y = KI / (sigma * np.sqrt(np.pi*a))
    logging.info(f"Geometry factor estimate: Y = KI / (sigma*sqrt(pi*a)) = {Y:.4f}")
    return Y


def export_line_csv(path: Path, data: Dict[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "r,x,y,sxx,syy,txy,von_mises\n"
    arr = np.column_stack([data["r"], data["x"], data["y"], data["sxx"], data["syy"], data["txy"], data["von_mises"]])
    np.savetxt(path, arr, delimiter=",", header=header, comments="")
    logging.info(f"Wrote CSV: {path}")

def write_summary_json(path: Path, summary: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Wrote summary: {path}")

# ----------------------------
# Main
# ----------------------------
def main():
    setup_logging()
    cfg = ValConfig()

    cfg.run_name = "meshrun_0.055mm"
    cfg.run_dir = Path("Data/New Data") / cfg.run_name

    cfg.vtu_path = cfg.run_dir / "solution_q4.vtu"
    cfg.npz_path = cfg.run_dir / "fields.npz"
    cfg.export_csv = cfg.run_dir / "validation_line.csv"
    manifest_path = cfg.run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = load_run_manifest(cfg.run_dir)
        manifest_hash = manifest.get("manifest_hash_sha256")
    else:
        _, manifest_hash = write_run_manifest(
            cfg.run_dir,
            {
                "workflow": "FEM.validate_fields",
                "geometry_mesh": {"a": cfg.a, "W": cfg.W, "H": cfg.H},
                "solver": {
                    "E": cfg.E,
                    "nu": cfg.nu,
                    "plane_stress": cfg.plane_stress,
                    "sigma_nominal": cfg.sigma_nominal,
                },
                "validation": {
                    "tip": list(cfg.tip),
                    "crack_dir": list(cfg.crack_dir),
                    "r_min": cfg.r_min,
                    "r_max": cfg.r_max,
                    "n_r": cfg.n_r,
                    "r_in": cfg.r_in,
                    "r_out_list": list(cfg.r_out_list),
                    "crack_face_exclusion": cfg.crack_face_exclusion,
                },
                "rng": {"seed_derivation_rule": "deterministic_no_rng"},
            },
        )

    outdir = cfg.run_dir
    outdir.mkdir(parents=True, exist_ok=True)

    pts, conn, u = load_solution(cfg)
    tip = np.array(cfg.tip, dtype=float)
    crack_dir = np.array(cfg.crack_dir, dtype=float)

    # ---- J sweep for KI vs r_out ----
    logging.info(f"Using crack_face_exclusion = {cfg.crack_face_exclusion:.3e} m")
    logging.info("Running J sweep for KI vs r_out ...")
    sweep = sweep_J_rout(
        pts=pts, conn=conn, U=u, tip=tip,
        E=cfg.E, nu=cfg.nu, plane_stress=cfg.plane_stress,
        r_in=cfg.r_in, r_out_list=list(cfg.r_out_list),
        crack_dir=crack_dir,
        log_each=True,
        crack_start=np.array([0.0, 0.0], dtype=float),
        crack_end=tip,
        exclude_crack_faces=True,
        crack_face_exclusion=cfg.crack_face_exclusion,
    )
    r_outs = [s.r_out for s in sweep]
    KIs = [s.KI for s in sweep]
    Js  = [s.J for s in sweep]

    J_rel_span = relative_span(Js)
    KI_rel_span = relative_span(KIs)

    logging.info(f"J relative span across contours  = {J_rel_span:.4%}")
    logging.info(f"KI relative span across contours = {KI_rel_span:.4%}")
    
    # choose a reference KI (use the middle r_out)
    best_idx = choose_best_plateau_index(Js)
    KI_ref = float(KIs[best_idx])
    J_ref = float(Js[best_idx])
    logging.info(f"Using plateau contour: r_out={r_outs[best_idx]:.4f}, " f"J={J_ref:.6e}, KI={KI_ref:.6e}")
   
    JK_resid = JK_consistency_residual(J_ref, KI_ref, cfg.E, cfg.nu, cfg.plane_stress)
    logging.info(f"J-K consistency residual = {JK_resid:.4e}")

    # ---- Y factor ----
    aW = cfg.a / cfg.W
    Y = print_Y_factor(KI_ref, cfg.sigma_nominal, cfg.a)
    logging.info(f"(for your geometry) a/W={aW:.4f}, Y_est={Y:.4f}")

    # ---- Sample stresses ahead of tip ----
    r = np.logspace(np.log10(cfg.r_min), np.log10(cfg.r_max), cfg.n_r)
    logging.info("Sampling stress fields along theta=0 ray ahead of crack tip ...")
    line = sample_stress_along_crack_line(
        pts=pts, conn=conn, u=u,
        tip=tip, crack_dir=crack_dir,
        r_vals=r,
        E=cfg.E, nu=cfg.nu, plane_stress=cfg.plane_stress
    )

    # ---- Plots ----
    outdir = cfg.run_dir
    outdir.mkdir(parents=True, exist_ok=True)

    plot_KI_vs_rout(r_outs, KIs, out_png=outdir / "KI_vs_rout.png")
    plot_sigma_sqrt_r(line["r"], line["syy"], KI_ref, out_png=outdir / "sigma_sqrt_2pir_vs_r.png")
    plot_sigma_and_vm(line["r"], line["syy"], line["von_mises"], out_png=outdir / "sigma_yy_and_vonmises_vs_r.png")
    plot_J_vs_rout(r_outs, Js, chosen_idx=best_idx, out_png=outdir / "J_path_independence.png")
    
    # ---- CSV export ----
    if cfg.export_csv is not None:
        export_line_csv(cfg.export_csv, line)
        
    summary = {
    "run_name": cfg.run_name,
    "a": cfg.a,
    "r_in": cfg.r_in,
    "r_out_list": list(r_outs),
    "J_list": list(map(float, Js)),
    "KI_list": list(map(float, KIs)),
    "best_idx": int(best_idx),
    "best_r_out": float(r_outs[best_idx]),
    "J_ref": float(J_ref),
    "KI_ref": float(KI_ref),
    "J_relative_span": float(J_rel_span),
    "KI_relative_span": float(KI_rel_span),
    "JK_relative_residual": float(JK_resid),
    "Y_est": float(Y),
    }
    write_summary_json(
        outdir / "validation_summary.json",
        ensure_summary_manifest_hash(summary, manifest_hash),
    )

    plt.show()


if __name__ == "__main__":
    main()

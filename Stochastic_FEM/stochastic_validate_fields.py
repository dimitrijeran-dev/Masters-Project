#!/usr/bin/env python3
"""
stochastic_validate_fields.py

Validation / postprocessing for stochastic fracture runs.

What this upgraded version adds
-------------------------------
- Uses the corrected interaction integral as the primary KI extractor
- Also computes corrected J* as a comparison / sanity check
- Supports stochastic elementwise E from either IID or KL random fields
- Aggregates per-realization KI statistics into a single summary JSON
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import sys

THIS = Path(__file__).resolve()
ROOT = THIS.parent
sys.path.append(str(ROOT))

try:
    from Stochastic_FEM.J_Integral_stochastic_modified import sweep_J_rout, sweep_interaction_rout
except Exception:
    from J_Integral_stochastic_modified import sweep_J_rout  # type: ignore
    sweep_interaction_rout = None  # type: ignore


@dataclass
class ValConfig:
    run_dir: Path = Path("Data/New Data/Smaller_rin_plate_edge_crack")
    realization_id: Optional[int] = None
    run_all_realizations: bool = True
    realization_glob: str = "fields_mc*.npz"

    E: float = 73.1e9
    nu: float = 0.33
    plane_stress: bool = True
    sigma_nominal: float = 50e6

    r_min: float = 2.0e-4
    r_max: float = 3.0e-2
    n_r: int = 25

    r_in: float = 0.008
    r_out_list: Tuple[float, ...] = (0.024, 0.026, 0.028, 0.032, 0.036, 0.038, 0.040)
    crack_face_exclusion: float = 5.0e-4

    export_csv: bool = True
    method_primary: str = "interaction"


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
        stem = p.stem
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
    N = np.array(
        [
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta),
        ],
        dtype=float,
    )
    dN_dxi = np.array(
        [
            [-0.25 * (1 - eta), -0.25 * (1 - xi)],
            [0.25 * (1 - eta), -0.25 * (1 + xi)],
            [0.25 * (1 + eta), 0.25 * (1 + xi)],
            [-0.25 * (1 + eta), 0.25 * (1 - xi)],
        ],
        dtype=float,
    )
    return N, dN_dxi


def D_matrix(E: float, nu: float, plane_stress: bool) -> np.ndarray:
    if plane_stress:
        c = E / (1.0 - nu ** 2)
        return c * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]], dtype=float)
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]], dtype=float)


def element_stress_at_parent_point(xe, ue, E, nu, plane_stress, xi, eta):
    _, dN_dxi = q4_shape(xi, eta)
    J = xe.T @ dN_dxi
    detJ = float(np.linalg.det(J))
    if detJ <= 0:
        raise ValueError("Non-positive detJ")
    invJ = np.linalg.inv(J)
    dN_dx = dN_dxi @ invJ.T

    grad_u = ue.T @ dN_dx
    eps = np.array([grad_u[0, 0], grad_u[1, 1], grad_u[0, 1] + grad_u[1, 0]], dtype=float)
    D = D_matrix(E, nu, plane_stress)
    sig = D @ eps
    sxx, syy, txy = sig
    vm = float(np.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * txy * txy))
    return sig, vm


def point_in_quad_newton(xe: np.ndarray, x_target: np.ndarray, tol=1e-10, maxiter=20):
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

    inside = (-1.0001 <= xi <= 1.0001) and (-1.0001 <= eta <= 1.0001)
    return inside, float(xi), float(eta)


def sample_stress_line(pts, conn, u, tip, crack_dir, r_vals, E_elem, E_scalar, nu, plane_stress):
    sigma_yy = np.full_like(r_vals, np.nan, dtype=float)
    von_mises = np.full_like(r_vals, np.nan, dtype=float)

    for i, r in enumerate(r_vals):
        xq = tip + r * crack_dir
        found = False
        for e, nodes in enumerate(conn):
            xe = pts[nodes, :]
            inside, xi, eta = point_in_quad_newton(xe, xq)
            if not inside:
                continue
            ue = np.column_stack([u[2 * nodes], u[2 * nodes + 1]])
            E_here = float(E_elem[e]) if E_elem is not None else float(E_scalar)
            sig, vm = element_stress_at_parent_point(xe, ue, E_here, nu, plane_stress, xi, eta)
            sigma_yy[i] = float(sig[1])
            von_mises[i] = vm
            found = True
            break
        if not found:
            sigma_yy[i] = np.nan
            von_mises[i] = np.nan

    return sigma_yy, von_mises


def plot_curve(x, y, xlabel, ylabel, title, out_png: Path, ref_y: Optional[float] = None, ref_label: str = "reference"):
    plt.figure()
    plt.plot(x, y, marker="o")
    if ref_y is not None and np.isfinite(ref_y):
        plt.axhline(ref_y, linestyle="--", label=ref_label)
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


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
    plt.axhline(KI_ref, linestyle="--", label=r"$K_I$ (primary)")
    plt.xscale("log")
    plt.xlabel("r ahead of tip (m)")
    plt.ylabel(r"$\sigma_{yy}\sqrt{2\pi r}$ (Pa*sqrt(m))")
    plt.title("LEFM near-tip check along crack line (theta=0)")
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


def choose_best_idx(vals: List[float]) -> int:
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    tail = arr[n // 2 :]
    ref = float(np.mean(tail))
    return int(np.argmin(np.abs(arr - ref)))


def relative_span(vals: List[float], ref_val: float) -> float:
    arr = np.asarray(vals, dtype=float)
    if abs(ref_val) < 1e-30:
        return float("nan")
    return float((np.max(arr) - np.min(arr)) / abs(ref_val))


def run_one_validation(cfg: ValConfig, realization_id: Optional[int]) -> dict:
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

    j_sweep = sweep_J_rout(
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
        use_eischen_correction=True,
    )
    j_routs = [float(s.r_out) for s in j_sweep]
    J_vals = [float(s.J) for s in j_sweep]
    KI_j_vals = [float(s.KI) for s in j_sweep]
    j_best_idx = choose_best_idx(KI_j_vals)

    KI_primary_vals = KI_j_vals
    primary_best_idx = j_best_idx
    KII_primary_vals = None

    if sweep_interaction_rout is not None:
        inter_sweep = sweep_interaction_rout(
            pts=pts,
            conn=conn,
            U=u,
            tip=tip,
            E=E_scalar,
            E_elem=E_elem,
            E_tip=E_scalar,
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
        inter_routs = [float(s.r_out) for s in inter_sweep]
        KI_inter_vals = [float(s.KI) for s in inter_sweep]
        KII_inter_vals = [float(s.KII) for s in inter_sweep]
        inter_best_idx = choose_best_idx(KI_inter_vals)

        if cfg.method_primary == "interaction":
            KI_primary_vals = KI_inter_vals
            KII_primary_vals = KII_inter_vals
            primary_best_idx = inter_best_idx
    else:
        inter_routs, KI_inter_vals, KII_inter_vals, inter_best_idx = [], [], [], None

    KI_ref = KI_primary_vals[primary_best_idx]
    KII_ref = KII_primary_vals[primary_best_idx] if KII_primary_vals is not None else None

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

    plot_curve(j_routs, J_vals, "r_out (m)", "J* (N/m)", "Corrected J* path independence", cfg.run_dir / f"J_star_path_independence{suffix}.png", ref_y=J_vals[j_best_idx], ref_label="chosen")
    plot_curve(j_routs, KI_j_vals, "r_out (m)", "K_I from J* (Pa*sqrt(m))", "K_I vs outer radius (corrected J*)", cfg.run_dir / f"KI_from_Jstar_vs_rout{suffix}.png", ref_y=KI_j_vals[j_best_idx], ref_label="chosen")

    if sweep_interaction_rout is not None:
        plot_curve(inter_routs, KI_inter_vals, "r_out (m)", "K_I from interaction integral (Pa*sqrt(m))", "K_I vs outer radius (interaction integral)", cfg.run_dir / f"KI_interaction_vs_rout{suffix}.png", ref_y=KI_inter_vals[inter_best_idx], ref_label="chosen")
        plot_curve(inter_routs, KII_inter_vals, "r_out (m)", "K_II from interaction integral (Pa*sqrt(m))", "K_II vs outer radius (interaction integral)", cfg.run_dir / f"KII_interaction_vs_rout{suffix}.png", ref_y=KII_inter_vals[inter_best_idx], ref_label="chosen")

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
        "r_in": cfg.r_in,
        "r_out_list": j_routs,
        "J_star_list": J_vals,
        "KI_from_J_star_list": KI_j_vals,
        "J_star_best_idx": j_best_idx,
        "J_star_ref": J_vals[j_best_idx],
        "KI_from_J_star_ref": KI_j_vals[j_best_idx],
        "J_star_relative_span": relative_span(J_vals, J_vals[j_best_idx]),
        "KI_from_J_star_relative_span": relative_span(KI_j_vals, KI_j_vals[j_best_idx]),
        "primary_method": cfg.method_primary,
        "KI_ref": KI_ref,
        "E_elem_mean": float(np.mean(E_elem)) if E_elem is not None else E_scalar,
        "E_elem_std": float(np.std(E_elem)) if E_elem is not None else 0.0,
    }

    if sweep_interaction_rout is not None:
        summary.update(
            {
                "interaction_r_out_list": inter_routs,
                "KI_interaction_list": KI_inter_vals,
                "KII_interaction_list": KII_inter_vals,
                "interaction_best_idx": inter_best_idx,
                "KI_interaction_ref": KI_inter_vals[inter_best_idx],
                "KII_interaction_ref": KII_inter_vals[inter_best_idx],
                "KI_interaction_relative_span": relative_span(KI_inter_vals, KI_inter_vals[inter_best_idx]),
                "KII_interaction_relative_span": relative_span(KII_inter_vals, KII_inter_vals[inter_best_idx]),
            }
        )
        if KII_ref is not None:
            summary["KII_ref"] = KII_ref

    (cfg.run_dir / f"validation_summary{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info(f"Finished validation for realization {realization_id}.")
    return summary


def main():
    setup_logging()
    cfg = ValConfig()

    if cfg.run_all_realizations:
        ids = discover_realization_ids(cfg.run_dir, cfg.realization_glob)
        if not ids:
            raise FileNotFoundError(f"No realization files found in {cfg.run_dir} matching {cfg.realization_glob}")

        logging.info(f"Found realization IDs: {ids}")
        summaries = [run_one_validation(cfg, rid) for rid in ids]
        agg = {
            "n_realizations": len(summaries),
            "primary_method": cfg.method_primary,
            "KI_ref_mean": float(np.mean([s["KI_ref"] for s in summaries])),
            "KI_ref_std": float(np.std([s["KI_ref"] for s in summaries])),
            "KI_ref_min": float(np.min([s["KI_ref"] for s in summaries])),
            "KI_ref_max": float(np.max([s["KI_ref"] for s in summaries])),
        }
        if all("KII_ref" in s for s in summaries):
            agg["KII_ref_mean"] = float(np.mean([s["KII_ref"] for s in summaries]))
            agg["KII_ref_std"] = float(np.std([s["KII_ref"] for s in summaries]))
        (cfg.run_dir / "validation_summary_all_realizations.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    else:
        run_one_validation(cfg, cfg.realization_id)


if __name__ == "__main__":
    main()

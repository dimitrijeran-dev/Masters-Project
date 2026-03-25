#!/usr/bin/env python3
"""
stochastic_fem_solver.py

A geometry-aware FEM solver with optional stochastic material parameters.

Supports:
- plate_edge_crack
- plate_hole_edge_crack

Stochastic material model:
- elementwise Young's modulus perturbation about E_mean
- bounded uniform or Gaussian-clipped perturbations
- optional Monte Carlo batch mode

Important
---------
This solver writes elementwise E_realization to cell data / metadata.
Your current J-integral utilities use a single scalar E. If you want a
strictly consistent stochastic J-integral, you should also modify src/J_Integral.py
to use elementwise constitutive matrices in the energy density.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class SolverConfig:
    base_out_dir: Path = Path("Data/New Data")
    run_name: Optional[str] = None
    run_dir: Optional[Path] = None

    geometry_type: str = "plate_hole_edge_crack"  # plate_edge_crack or plate_hole_edge_crack

    # Geometry
    W: float = 1
    H: float = 0.5
    a: float = 0.040
    crack_gap: float = 5e-5
    hole_radius: float = 0.010
    hole_center: Optional[Tuple[float, float]] = None

    # Material
    E_mean: float = 73.1e9
    nu: float = 0.33
    plane_stress: bool = True
    thickness: float = 0.002

    # Stochastic material settings
    stochastic_E: bool = True
    E_mode: str = "uniform_bounded"  # uniform_bounded or gaussian_clipped
    E_rel_std: float = 0.005
    E_rel_clip: float = 0.005
    random_seed: int = 5
    n_realizations: int = 5

    # Boundary conditions / loading
    fix_bottom_uy: bool = True
    traction_top: Tuple[float, float] = (0.0, +50e6)
    traction_bottom: Tuple[float, float] = (0.0, 0.0)

    # File names
    msh_name: str = "mesh_q4.msh"
    vtk_name: str = "solution_q4.vtk"
    npz_name: str = "fields.npz"
    meta_name: str = "metadata.json"


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def crack_tip_xy(cfg: SolverConfig) -> Tuple[float, float]:
    if cfg.geometry_type == "plate_edge_crack":
        return (cfg.a, 0.0)
    if cfg.geometry_type == "plate_hole_edge_crack":
        cx, cy = cfg.hole_center if cfg.hole_center is not None else (cfg.W / 2.0, 0.0)
        return (cx + cfg.hole_radius + cfg.a, cy)
    raise ValueError(f"Unsupported geometry_type={cfg.geometry_type!r}")


def crack_start_xy(cfg: SolverConfig) -> Tuple[float, float]:
    if cfg.geometry_type == "plate_edge_crack":
        return (0.0, 0.0)
    if cfg.geometry_type == "plate_hole_edge_crack":
        cx, cy = cfg.hole_center if cfg.hole_center is not None else (cfg.W / 2.0, 0.0)
        return (cx + cfg.hole_radius, cy)
    raise ValueError(f"Unsupported geometry_type={cfg.geometry_type!r}")


def read_gmsh_quad_mesh(msh_path: Path):
    m = meshio.read(str(msh_path))
    pts = np.asarray(m.points[:, :2], dtype=float)

    quad_blocks = []
    line_blocks = []
    line_phys_blocks = []

    phys_blocks = None
    if m.cell_data and "gmsh:physical" in m.cell_data:
        phys_blocks = m.cell_data["gmsh:physical"]

    for i, block in enumerate(m.cells):
        if block.type in ("quad", "quad4", "quadrilateral"):
            quad_blocks.append(np.asarray(block.data, dtype=int))

        elif block.type == "line":
            line_blocks.append(np.asarray(block.data, dtype=int))
            if phys_blocks is not None and i < len(phys_blocks):
                arr = np.asarray(phys_blocks[i], dtype=int)
                if len(arr) == len(block.data):
                    line_phys_blocks.append(arr)

    if not quad_blocks:
        raise ValueError("No quad cells found.")

    quad = np.vstack(quad_blocks)
    lines = np.vstack(line_blocks) if line_blocks else None
    line_phys = np.concatenate(line_phys_blocks) if line_phys_blocks else None

    phys_map: Dict[int, str] = {}
    if hasattr(m, "field_data") and m.field_data:
        for name, (pid, dim) in m.field_data.items():
            phys_map[int(pid)] = name

    logging.info(f"Read mesh: {msh_path}")
    logging.info(f"Cell blocks: {[blk.type for blk in m.cells]}")
    logging.info(f"Physical groups found: {phys_map}")
    logging.info(
        f"lines={'yes' if lines is not None else 'no'}, "
        f"line_phys={'yes' if line_phys is not None else 'no'}"
    )

    return pts, quad, lines, line_phys, phys_map

def boundary_edges_from_geometry(
    pts: np.ndarray,
    quad: np.ndarray,
    name: str,
    W: float,
    H: float,
) -> np.ndarray:
    """
    Fallback boundary-edge extraction directly from the quad mesh topology.

    Returns unique edges lying on:
      TOP    -> y = +H/2
      BOTTOM -> y = -H/2
      LEFT   -> x = 0
      RIGHT  -> x = W
    """
    tol = 1e-8 * max(W, H)

    edge_count = {}

    for elem in quad:
        # local quad edges
        local_edges = [
            (elem[0], elem[1]),
            (elem[1], elem[2]),
            (elem[2], elem[3]),
            (elem[3], elem[0]),
        ]
        for i, j in local_edges:
            e = tuple(sorted((int(i), int(j))))
            edge_count[e] = edge_count.get(e, 0) + 1

    # boundary edges appear only once
    boundary_edges = [e for e, c in edge_count.items() if c == 1]

    selected = []
    for i, j in boundary_edges:
        x1, y1 = pts[i]
        x2, y2 = pts[j]

        if name == "TOP":
            if abs(y1 - H / 2.0) < tol and abs(y2 - H / 2.0) < tol:
                selected.append([i, j])

        elif name == "BOTTOM":
            if abs(y1 + H / 2.0) < tol and abs(y2 + H / 2.0) < tol:
                selected.append([i, j])

        elif name == "LEFT":
            if abs(x1 - 0.0) < tol and abs(x2 - 0.0) < tol:
                selected.append([i, j])

        elif name == "RIGHT":
            if abs(x1 - W) < tol and abs(x2 - W) < tol:
                selected.append([i, j])

        else:
            raise ValueError(f"Unsupported boundary name {name!r}")

    arr = np.asarray(selected, dtype=int)
    if arr.size == 0:
        raise ValueError(f"Could not infer boundary '{name}' from quad geometry.")
    return arr


def boundary_edges_by_name(
    pts: np.ndarray,
    quad: np.ndarray,
    lines,
    line_phys,
    phys_map,
    name: str,
    W: float,
    H: float,
) -> np.ndarray:
    """
    Try physical-group lookup first.
    If line tags are missing, fall back to geometric extraction from quads.
    """
    if lines is not None and line_phys is not None:
        pid = next((k for k, v in phys_map.items() if v == name), None)
        if pid is not None:
            edges = lines[line_phys == pid]
            if len(edges) > 0:
                logging.info(f"Using physical-group boundary '{name}' with {len(edges)} edges.")
                return edges

    logging.warning(
        f"Physical-tag lookup failed for boundary '{name}'. "
        f"Falling back to geometric boundary detection."
    )
    edges = boundary_edges_from_geometry(pts, quad, name, W, H)
    logging.info(f"Using geometric boundary '{name}' with {len(edges)} edges.")
    return edges


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


def q4_B_matrix(xe: np.ndarray, xi: float, eta: float):
    _, dN_dxi = q4_shape(xi, eta)
    J = xe.T @ dN_dxi
    detJ = float(np.linalg.det(J))
    if detJ <= 0:
        raise ValueError(f"Non-positive detJ={detJ}")
    invJ = np.linalg.inv(J)
    dN_dx = dN_dxi @ invJ.T

    B = np.zeros((3, 8), dtype=float)
    for a in range(4):
        dNdx, dNdy = dN_dx[a, 0], dN_dx[a, 1]
        B[0, 2*a] = dNdx
        B[1, 2*a+1] = dNdy
        B[2, 2*a] = dNdy
        B[2, 2*a+1] = dNdx
    return B, detJ


def sample_elementwise_E(cfg: SolverConfig, ne: int, rng: np.random.Generator) -> np.ndarray:
    if not cfg.stochastic_E:
        return np.full(ne, cfg.E_mean, dtype=float)

    if cfg.E_mode == "uniform_bounded":
        rel = rng.uniform(-cfg.E_rel_std, cfg.E_rel_std, size=ne)
    elif cfg.E_mode == "gaussian_clipped":
        rel = rng.normal(0.0, cfg.E_rel_std, size=ne)
        rel = np.clip(rel, -cfg.E_rel_clip, cfg.E_rel_clip)
    else:
        raise ValueError(f"Unsupported E_mode={cfg.E_mode!r}")

    return cfg.E_mean * (1.0 + rel)


def assemble_global_K(cfg: SolverConfig, pts: np.ndarray, quad: np.ndarray, E_elem: np.ndarray) -> sp.csr_matrix:
    n = pts.shape[0]
    ndof = 2 * n
    g = 1.0 / np.sqrt(3.0)
    gps = [(-g, -g), (g, -g), (g, g), (-g, g)]
    rows, cols, vals = [], [], []
    t = cfg.thickness

    for e in range(quad.shape[0]):
        nodes = quad[e]
        xe = pts[nodes, :]
        D = D_matrix(float(E_elem[e]), cfg.nu, cfg.plane_stress)

        Ke = np.zeros((8, 8), dtype=float)
        for (xi, eta) in gps:
            B, detJ = q4_B_matrix(xe, xi, eta)
            Ke += (B.T @ D @ B) * (detJ * t)

        dofs = np.array([2*nodes[0], 2*nodes[0]+1,
                         2*nodes[1], 2*nodes[1]+1,
                         2*nodes[2], 2*nodes[2]+1,
                         2*nodes[3], 2*nodes[3]+1], dtype=int)
        for i_local in range(8):
            for j_local in range(8):
                rows.append(dofs[i_local])
                cols.append(dofs[j_local])
                vals.append(Ke[i_local, j_local])

    return sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()


def assemble_uniform_traction_rhs(cfg: SolverConfig, pts: np.ndarray, boundary_edges: np.ndarray, tx: float, ty: float) -> np.ndarray:
    n = pts.shape[0]
    f = np.zeros(2 * n, dtype=float)
    tvec = np.array([tx, ty], dtype=float)
    for (n1, n2) in boundary_edges:
        L = float(np.linalg.norm(pts[n2] - pts[n1]))
        fe = cfg.thickness * L * 0.5 * tvec
        f[2*n1:2*n1+2] += fe
        f[2*n2:2*n2+2] += fe
    return f


def solve_dirichlet_elimination(K: sp.csr_matrix, f: np.ndarray, fixed_dofs: np.ndarray, fixed_vals: np.ndarray) -> np.ndarray:
    ndof = K.shape[0]
    all_dofs = np.arange(ndof, dtype=int)
    free = np.setdiff1d(all_dofs, fixed_dofs)
    f_mod = f.copy()
    f_mod[free] -= K[free][:, fixed_dofs] @ fixed_vals
    uf = spla.spsolve(K[free][:, free].tocsr(), f_mod[free])

    u = np.zeros(ndof, dtype=float)
    u[free] = uf
    u[fixed_dofs] = fixed_vals
    return u


def compute_cell_stress_vonmises(cfg: SolverConfig, pts: np.ndarray, quad: np.ndarray, u: np.ndarray, E_elem: np.ndarray):
    g = 1.0 / np.sqrt(3.0)
    gps = [(-g, -g), (g, -g), (g, g), (-g, g)]

    sig_cell = np.zeros((quad.shape[0], 3), dtype=float)
    vm_cell = np.zeros(quad.shape[0], dtype=float)

    for e in range(quad.shape[0]):
        nodes = quad[e]
        xe = pts[nodes, :]
        ue = np.column_stack([u[2*nodes], u[2*nodes + 1]])
        D = D_matrix(float(E_elem[e]), cfg.nu, cfg.plane_stress)

        sig_acc = np.zeros(3, dtype=float)
        vm_acc = 0.0
        wt_acc = 0.0

        for (xi, eta) in gps:
            _, dN_dxi = q4_shape(xi, eta)
            J = xe.T @ dN_dxi
            detJ = float(np.linalg.det(J))
            invJ = np.linalg.inv(J)
            dN_dx = dN_dxi @ invJ.T
            grad_u = ue.T @ dN_dx
            exx = grad_u[0, 0]
            eyy = grad_u[1, 1]
            gxy = grad_u[0, 1] + grad_u[1, 0]
            eps = np.array([exx, eyy, gxy], dtype=float)
            sig = D @ eps
            sxx, syy, txy = float(sig[0]), float(sig[1]), float(sig[2])
            vm = float(np.sqrt(sxx*sxx - sxx*syy + syy*syy + 3.0*txy*txy))
            wt = detJ
            sig_acc += sig * wt
            vm_acc += vm * wt
            wt_acc += wt

        sig_cell[e, :] = sig_acc / wt_acc
        vm_cell[e] = vm_acc / wt_acc

    return sig_cell, vm_cell


def write_vtk(out_path: Path, pts: np.ndarray, quad: np.ndarray, u: np.ndarray, sig_cell: np.ndarray, vm_cell: np.ndarray, E_elem: np.ndarray):
    pts3 = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(pts.shape[0])])
    U3 = np.column_stack([u[0::2], u[1::2], np.zeros(pts.shape[0])])
    mesh = meshio.Mesh(
        points=pts3,
        cells=[("quad", quad)],
        point_data={"U": U3},
        cell_data={
            "sigma": [sig_cell],
            "von_mises": [vm_cell],
            "E_realization": [E_elem],
            "sigma_xx": [sig_cell[:, 0]],
            "sigma_yy": [sig_cell[:, 1]],
            "tau_xy": [sig_cell[:, 2]],
        }
    )
    mesh.write(str(out_path))


def solve_one(cfg: SolverConfig, realization_id: int = 0):
    pts, quad, lines, line_phys, phys_map = read_gmsh_quad_mesh(cfg.run_dir / cfg.msh_name)
    top_edges = boundary_edges_by_name(
    pts, quad, lines, line_phys, phys_map, "TOP", cfg.W, cfg.H
)
    bottom_edges = boundary_edges_by_name(
    pts, quad, lines, line_phys, phys_map, "BOTTOM", cfg.W, cfg.H
)

    rng = np.random.default_rng(cfg.random_seed + realization_id)
    E_elem = sample_elementwise_E(cfg, quad.shape[0], rng)

    K = assemble_global_K(cfg, pts, quad, E_elem)
    f = assemble_uniform_traction_rhs(cfg, pts, top_edges, cfg.traction_top[0], cfg.traction_top[1])
    f += assemble_uniform_traction_rhs(cfg, pts, bottom_edges, cfg.traction_bottom[0], cfg.traction_bottom[1])

    bottom_nodes = np.unique(bottom_edges.reshape(-1))
    fixed_dofs = list((2 * bottom_nodes + 1).astype(int))
    fixed_vals = list(np.zeros_like(bottom_nodes, dtype=float))

    x_bottom = pts[bottom_nodes, 0]
    x_center = 0.5 * (np.min(x_bottom) + np.max(x_bottom))
    anchor = int(bottom_nodes[np.argmin(np.abs(x_bottom - x_center))])
    fixed_dofs.append(2 * anchor)
    fixed_vals.append(0.0)
    fixed_dofs = np.array(fixed_dofs, dtype=int)
    fixed_vals = np.array(fixed_vals, dtype=float)

    u = solve_dirichlet_elimination(K, f, fixed_dofs, fixed_vals)
    sig_cell, vm_cell = compute_cell_stress_vonmises(cfg, pts, quad, u, E_elem)

    suffix = f"_mc{realization_id:04d}" if cfg.n_realizations > 1 else ""
    write_vtk(cfg.run_dir / f"solution_q4{suffix}.vtk", pts, quad, u, sig_cell, vm_cell, E_elem)
    np.savez(cfg.run_dir / f"fields{suffix}.npz", pts=pts, conn=quad, u=u, E_elem=E_elem)

    meta = asdict(cfg)
    for k, v in list(meta.items()):
        if isinstance(v, Path):
            meta[k] = str(v)

    meta["tip"] = list(crack_tip_xy(cfg))
    meta["crack_start"] = list(crack_start_xy(cfg))
    meta["crack_dir"] = [1.0, 0.0]
    meta["realization_id"] = int(realization_id)
    meta["E_elem_mean"] = float(np.mean(E_elem))
    meta["E_elem_std"] = float(np.std(E_elem))

    (cfg.run_dir / f"metadata{suffix}.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8"
    )

    logging.info(f"Finished realization {realization_id}: mean(E)={np.mean(E_elem):.6e}, std(E)={np.std(E_elem):.6e}")


def main():
    setup_logging()
    cfg = SolverConfig()
    if cfg.run_name is None:
        cfg.run_name = f"stoch_{cfg.geometry_type}"
    cfg.run_dir = cfg.base_out_dir / cfg.run_name
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    # User should generate mesh first with stochastic_mesh.py into the same run_dir
    msh_path = cfg.run_dir / cfg.msh_name
    if not msh_path.exists():
        raise FileNotFoundError(f"Missing mesh file: {msh_path}. Run stochastic_mesh.py first.")

    for i in range(cfg.n_realizations):
        solve_one(cfg, i)


if __name__ == "__main__":
    main()

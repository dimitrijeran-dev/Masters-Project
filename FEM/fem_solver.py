#!/usr/bin/env python3
"""
fem_solver.py

- Reads a Gmsh .msh file (unstructured quads) via meshio
- Assembles Q4 plane stress/strain stiffness (2x2 Gauss)
- Applies uniform traction on TOP boundary
- Fixes the BOTTOM boundary (u=v=0)
- Writes solution to VTK for ParaView

Key fix vs earlier versions:
- Gmsh .msh often contains MULTIPLE "line" cell blocks. We now concatenate ALL of them and
  their corresponding gmsh:physical tags, so boundary extraction works and RHS is nonzero.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import meshio
from typing import Optional

import json

import sys
from pathlib import Path

# Add project root (Masters-Project/) to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.configs.geometry import geometry_payload
from src.configs.material import material_payload
from src.configs.run_io import load_runtime_config, update_runtime_config

# ----------------------------
# Config
# ----------------------------
@dataclass
class SolverConfig:
    base_out_dir: Path = Path("Data/New Data")
    run_name: Optional[str] = None

    # these get assigned in main()
    run_dir: Optional[Path] = None
    msh_path: Optional[Path] = None
    out_vtk: Optional[Path] = None
    out_npz: Optional[Path] = None
    meta_path: Optional[Path] = None


    # Geometry (used only for optional coordinate fallback / sanity)
    W: float = 0.200
    H: float = 0.100
    a: float = 0.055
    crack_gap: float = 5e-5

    # Material
    E: float = 73.1e9
    nu: float = 0.33
    plane_stress: bool = True
    thickness: float = 0.002

    # Boundary conditions
    fix_bottom_uy: bool = True

    # Top traction only: vertical tensile loading
    traction_top: Tuple[float, float] = (0.0, +50e6)
    traction_bottom: Tuple[float, float] = (0.0, 0.0)   # no bottom traction
    traction: float = 50e6


# ----------------------------
# Logging
# ----------------------------
def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ----------------------------
# Mesh read helpers
# ----------------------------
def read_gmsh_quad_mesh(msh_path: Path):
    """
    Returns:
      pts: (N,2)
      quad: (ne,4)
      lines: (nb,2) or None
      line_phys: (nb,) or None  physical ids for each line segment
      phys_map: dict[int,str] mapping physical id -> name
    """
    m = meshio.read(str(msh_path))
    pts = np.asarray(m.points[:, :2], dtype=float)

    # -------- quads (concatenate all quad blocks, not just first) --------
    quad_blocks = []
    for block in m.cells:
        if block.type in ("quad", "quad4", "quadrilateral"):
            quad_blocks.append(np.asarray(block.data, dtype=int))
    if not quad_blocks:
        raise ValueError("No quad elements found in mesh. Ensure Gmsh recombination produced quads.")
    quad = np.vstack(quad_blocks)

    # -------- lines + physical tags (concatenate ALL line blocks) --------
    lines_blocks = []
    line_phys_blocks = []

    phys_blocks = None
    if m.cell_data and "gmsh:physical" in m.cell_data:
        phys_blocks = m.cell_data["gmsh:physical"]

    for i, block in enumerate(m.cells):
        if block.type == "line":
            lines_blocks.append(np.asarray(block.data, dtype=int))
            if phys_blocks is not None and i < len(phys_blocks):
                line_phys_blocks.append(np.asarray(phys_blocks[i], dtype=int))

    lines = np.vstack(lines_blocks) if lines_blocks else None
    line_phys = np.concatenate(line_phys_blocks) if line_phys_blocks else None

    # -------- physical id -> name map --------
    phys_map: Dict[int, str] = {}
    if hasattr(m, "field_data") and m.field_data:
        # field_data[name] = [id, dim]
        for name, (pid, dim) in m.field_data.items():
            phys_map[int(pid)] = name

    return pts, quad, lines, line_phys, phys_map


def boundary_edges_by_name(
    lines: np.ndarray,
    line_phys: np.ndarray,
    phys_map: Dict[int, str],
    name: str
) -> np.ndarray:
    if lines is None or line_phys is None:
        raise ValueError("Boundary line elements or gmsh:physical tags missing from .msh.")
    pid = None
    for k, v in phys_map.items():
        if v == name:
            pid = k
            break
    if pid is None:
        raise ValueError(f"Physical group '{name}' not found. Available: {sorted(set(phys_map.values()))}")
    return lines[line_phys == pid]


def boundary_edges_by_coord(
    lines: np.ndarray,
    pts: np.ndarray,
    which: str,
    W: float,
    H: float,
    tol: float = 1e-8
) -> np.ndarray:
    """
    Coordinate-based fallback if physical tags are missing or misbehave.
    which ∈ {"RIGHT","LEFT","TOP","BOTTOM"}
    """
    edges = []
    for (n1, n2) in lines:
        x1, y1 = pts[n1, 0], pts[n1, 1]
        x2, y2 = pts[n2, 0], pts[n2, 1]

        if which == "RIGHT":
            if abs(x1 - W) < tol and abs(x2 - W) < tol:
                edges.append([n1, n2])
        elif which == "LEFT":
            if abs(x1 - 0.0) < tol and abs(x2 - 0.0) < tol:
                edges.append([n1, n2])
        elif which == "TOP":
            if abs(y1 - (H / 2.0)) < tol and abs(y2 - (H / 2.0)) < tol:
                edges.append([n1, n2])
        elif which == "BOTTOM":
            if abs(y1 - (-H / 2.0)) < tol and abs(y2 - (-H / 2.0)) < tol:
                edges.append([n1, n2])
        else:
            raise ValueError("which must be RIGHT/LEFT/TOP/BOTTOM")

    return np.array(edges, dtype=int)


# ----------------------------
# Q4 element routines
# ----------------------------
def D_matrix(E: float, nu: float, plane_stress: bool) -> np.ndarray:
    if plane_stress:
        c = E / (1.0 - nu**2)
        return c * np.array([[1.0, nu, 0.0],
                             [nu, 1.0, 0.0],
                             [0.0, 0.0, (1.0 - nu) / 2.0]], dtype=float)
    # plane strain
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([[1.0 - nu, nu, 0.0],
                         [nu, 1.0 - nu, 0.0],
                         [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]], dtype=float)


def q4_shape(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      N: (4,)
      dN_dxi: (4,2) columns [dN/dxi, dN/deta]
    """
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


def q4_B_matrix(xe: np.ndarray, xi: float, eta: float) -> Tuple[np.ndarray, float]:
    """
    xe: (4,2) nodal coords.
    Returns:
      B: (3,8)
      detJ: float
    """
    _, dN_dxi = q4_shape(xi, eta)          # (4,2)
    J = xe.T @ dN_dxi                      # (2,2)
    detJ = float(np.linalg.det(J))
    if detJ <= 0:
        raise ValueError(f"Non-positive detJ={detJ}. Check element orientation / mesh quality.")
    invJ = np.linalg.inv(J)
    dN_dx = dN_dxi @ invJ.T                # (4,2)

    B = np.zeros((3, 8), dtype=float)
    for a in range(4):
        dNdx, dNdy = dN_dx[a, 0], dN_dx[a, 1]
        B[0, 2 * a]     = dNdx
        B[1, 2 * a + 1] = dNdy
        B[2, 2 * a]     = dNdy
        B[2, 2 * a + 1] = dNdx

    return B, detJ

def q4_B_and_grad_u(xe: np.ndarray, ue: np.ndarray, xi: float, eta: float):
    """
    Returns:
      B: (3,8)
      detJ: float
      grad_u: (2,2) [[du/dx, du/dy],
                    [dv/dx, dv/dy]]
    """
    N, dN_dxi = q4_shape(xi, eta)          # (4,), (4,2)
    J = xe.T @ dN_dxi                      # (2,2)
    detJ = float(np.linalg.det(J))
    if detJ <= 0:
        raise ValueError(f"Non-positive detJ={detJ}")
    invJ = np.linalg.inv(J)
    dN_dx = dN_dxi @ invJ.T                # (4,2)

    # grad_u = ue^T * dN_dx
    grad_u = ue.T @ dN_dx                  # (2,2)

    B = np.zeros((3, 8), dtype=float)
    for a in range(4):
        dNdx, dNdy = dN_dx[a, 0], dN_dx[a, 1]
        B[0, 2*a]     = dNdx
        B[1, 2*a+1]   = dNdy
        B[2, 2*a]     = dNdy
        B[2, 2*a+1]   = dNdx

    return B, detJ, grad_u


def von_mises_from_sig(sxx: float, syy: float, txy: float, plane_stress: bool, nu: float, exx: float, eyy: float) -> float:
    """
    von Mises stress.
    - Plane stress: szz = 0
    - Plane strain: szz = nu*(sxx + syy) is NOT generally exact; better is szz = λ (exx+eyy) with εzz=0.
      We'll compute szz from Lamé constants using strains (more correct).
    """
    if plane_stress:
        szz = 0.0
    else:
        # plane strain: εzz = 0 => σzz = λ (εxx + εyy)
        E = 1.0  # dummy; we don't have E here; compute using nu only isn't possible.
        # We'll compute σzz in the caller where E is known, OR pass E in.
        raise RuntimeError("For plane_strain von Mises, use von_mises_plane_strain(E, nu, exx, eyy, sxx, syy, txy).")

    return float(np.sqrt(sxx*sxx - sxx*syy + syy*syy + 3.0*txy*txy))


def von_mises_plane_strain(E: float, nu: float, exx: float, eyy: float, sxx: float, syy: float, txy: float) -> float:
    """
    3D von Mises using σzz from plane strain (εzz=0):
      σzz = λ(εxx+εyy)
    """
    lam = E*nu/((1.0+nu)*(1.0-2.0*nu))
    szz = lam*(exx + eyy)
    return float(np.sqrt(0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2) + 3.0*(txy**2)))


def compute_cell_stress_vonmises(cfg, pts: np.ndarray, quad: np.ndarray, u: np.ndarray):
    """
    Computes per-element averaged stress (sxx,syy,txy) and von Mises.
    Returns:
      sig_cell: (ne,3)
      vm_cell:  (ne,)
    """
    D = D_matrix(cfg.E, cfg.nu, cfg.plane_stress)

    # 2x2 Gauss
    g = 1.0/np.sqrt(3.0)
    gps = [(-g, -g), ( g, -g), ( g,  g), (-g,  g)]
    w = 1.0

    ne = quad.shape[0]
    sig_cell = np.zeros((ne, 3), dtype=float)
    vm_cell  = np.zeros(ne, dtype=float)

    for e in range(ne):
        nodes = quad[e]
        xe = pts[nodes, :]
        ue = np.column_stack([u[2*nodes], u[2*nodes + 1]])  # (4,2)

        sig_acc = np.zeros(3, dtype=float)
        vm_acc = 0.0
        wt_acc = 0.0

        for (xi, eta) in gps:
            B, detJ, grad_u = q4_B_and_grad_u(xe, ue, xi, eta)

            # strain Voigt: [exx, eyy, gxy], gxy = du/dy + dv/dx
            exx = grad_u[0, 0]
            eyy = grad_u[1, 1]
            gxy = grad_u[0, 1] + grad_u[1, 0]
            eps = np.array([exx, eyy, gxy], dtype=float)

            sig = D @ eps  # [sxx, syy, txy]
            sxx, syy, txy = float(sig[0]), float(sig[1]), float(sig[2])

            if cfg.plane_stress:
                vm = float(np.sqrt(sxx*sxx - sxx*syy + syy*syy + 3.0*txy*txy))
            else:
                vm = von_mises_plane_strain(cfg.E, cfg.nu, exx, eyy, sxx, syy, txy)

            wt = detJ * w * w
            sig_acc += sig * wt
            vm_acc  += vm  * wt
            wt_acc  += wt

        sig_cell[e, :] = sig_acc / wt_acc
        vm_cell[e]     = vm_acc  / wt_acc

    return sig_cell, vm_cell


# ----------------------------
# Assembly
# ----------------------------
def assemble_global_K(cfg: SolverConfig, pts: np.ndarray, quad: np.ndarray) -> sp.csr_matrix:
    n = pts.shape[0]
    ndof = 2 * n
    D = D_matrix(cfg.E, cfg.nu, cfg.plane_stress)

    # 2x2 Gauss
    g = 1.0 / np.sqrt(3.0)
    gps = [(-g, -g), (g, -g), (g, g), (-g, g)]
    w = 1.0

    rows, cols, vals = [], [], []
    t = cfg.thickness

    logging.info("Assembling global K (Q4, 2x2 Gauss)...")
    for e in range(quad.shape[0]):
        nodes = quad[e]
        xe = pts[nodes, :]  # (4,2)

        Ke = np.zeros((8, 8), dtype=float)
        for (xi, eta) in gps:
            B, detJ = q4_B_matrix(xe, xi, eta)
            Ke += (B.T @ D @ B) * (detJ * w * w * t)

        dofs = np.array([
            2 * nodes[0], 2 * nodes[0] + 1,
            2 * nodes[1], 2 * nodes[1] + 1,
            2 * nodes[2], 2 * nodes[2] + 1,
            2 * nodes[3], 2 * nodes[3] + 1
        ], dtype=int)

        for i_local in range(8):
            ii = dofs[i_local]
            for j_local in range(8):
                rows.append(ii)
                cols.append(dofs[j_local])
                vals.append(Ke[i_local, j_local])

    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    logging.info(f"K: shape={K.shape}, nnz={K.nnz}")
    return K


def assemble_uniform_traction_rhs(
    cfg: SolverConfig,
    pts: np.ndarray,
    boundary_edges: np.ndarray,
    tx: float,
    ty: float
) -> np.ndarray:
    """
    Consistent nodal force for uniform traction on line edges:
      fe = ∫ N^T t ds = (L/2)[t; t]
    """
    n = pts.shape[0]
    f = np.zeros(2 * n, dtype=float)
    tvec = np.array([tx, ty], dtype=float)

    for (n1, n2) in boundary_edges:
        x1 = pts[n1]
        x2 = pts[n2]
        L = float(np.linalg.norm(x2 - x1))
        fe = cfg.thickness * L * 0.5 * tvec
        f[2 * n1:2 * n1 + 2] += fe
        f[2 * n2:2 * n2 + 2] += fe

    return f


def validate_solution(cfg: SolverConfig,
                    pts: np.ndarray,
                    u: np.ndarray,
                    K: sp.csr_matrix,
                    f: np.ndarray,
                    top_edges: np.ndarray,
                    bottom_edges: np.ndarray):
    """
    Expert-style sanity checks:
    (1) Total applied forces on TOP/BOTTOM from assembled RHS
    (2) Reaction forces on fixed RIGHT boundary
    (3) Far-field strain estimate from uy(y) at x ~ x_probe
    """

    # ---- (1) Applied forces from RHS ----
    Fy_total = float(np.sum(f[1::2]))
    Fx_total = float(np.sum(f[0::2]))

    # Force contributed only by top/bottom (recompute quickly from edges using your traction values)
    # Note: this matches your assembly formula exactly.
    def traction_total(edges, tx, ty):
        F = np.zeros(2, dtype=float)
        for (n1, n2) in edges:
            L = float(np.linalg.norm(pts[n2] - pts[n1]))
            # each edge contributes thickness * traction * L (since fe_node is L/2 per node, two nodes => L)
            F += cfg.thickness * L * np.array([tx, ty], dtype=float)
        return F

    F_top = traction_total(top_edges, cfg.traction_top[0], cfg.traction_top[1])
    F_bot = traction_total(bottom_edges, cfg.traction_bottom[0], cfg.traction_bottom[1])
    F_ext = F_top + F_bot

    logging.info("---- Validation: Applied forces ----")
    logging.info(f"Sum(fx) = {Fx_total:.6e} N, Sum(fy) = {Fy_total:.6e} N  (global RHS check)")
    logging.info(f"Top traction total force    = [{F_top[0]:.6e}, {F_top[1]:.6e}] N")
    logging.info(f"Bottom traction total force = [{F_bot[0]:.6e}, {F_bot[1]:.6e}] N")
    logging.info(f"Total external force        = [{F_ext[0]:.6e}, {F_ext[1]:.6e}] N")

    # Compare to nominal sigma*W*t for the Y component (use abs because directions differ)
    sigma_nom = abs(cfg.traction_top[1]) if abs(cfg.traction_top[1]) > 0 else abs(cfg.traction_bottom[1])
    Fy_nom = sigma_nom * cfg.W * cfg.thickness
    logging.info(f"Nominal |Fy| per edge ~ sigma*W*t = {Fy_nom:.6e} N  (sigma={sigma_nom:.3e} Pa)")

    r = K @ u - f
    bottom_nodes = np.unique(bottom_edges.reshape(-1))
    bottom_dofs_x = 2 * bottom_nodes
    bottom_dofs_y = 2 * bottom_nodes + 1
    Rx_bottom = float(np.sum(r[bottom_dofs_x]))
    Ry_bottom = float(np.sum(r[bottom_dofs_y]))

    logging.info("---- Validation: Reactions on BOTTOM support ----")
    logging.info(f"Sum reactions on BOTTOM: Rx = {Rx_bottom:.6e} N, Ry = {Ry_bottom:.6e} N")

    # ---- (3) Far-field strain estimate from uy(y) at a probe x ----
    # Choose probe line near the right, but not on the clamp (avoid x=W exactly).
    x_probe = 0.85 * cfg.W
    band = 0.02 * cfg.W  # 2% of width band
    mask = np.abs(pts[:, 0] - x_probe) < band
    y = pts[mask, 1]
    uy = u[1::2][mask]

    logging.info("---- Validation: Far-field strain estimate ----")
    if y.size < 20:
        logging.warning("Not enough nodes found in probe band for strain fit. Increase band or adjust x_probe.")
        return

    # Fit uy = a*y + b
    A = np.column_stack([y, np.ones_like(y)])
    a, b = np.linalg.lstsq(A, uy, rcond=None)[0]
    eps_yy_fit = float(a)

    # Compare to sigma/E (plane stress)
    eps_yy_nom = float(sigma_nom / cfg.E) if cfg.plane_stress else float(sigma_nom * (1 - cfg.nu**2) / cfg.E)

    logging.info(f"Fitted eps_yy (from uy vs y) ~ {eps_yy_fit:.6e}")
    logging.info(f"Nominal eps_yy ~ sigma/E     ~ {eps_yy_nom:.6e}  (plane_stress={cfg.plane_stress})")
    logging.info(f"Ratio (fit/nominal)          = {eps_yy_fit/eps_yy_nom:.3f}")

    # Also estimate far-field separation from fit: Delta uy across height H
    dUy_fit = eps_yy_fit * cfg.H
    dUy_nom = eps_yy_nom * cfg.H
    logging.info(f"Delta uy across H (fit)      ~ {dUy_fit:.6e} m")
    logging.info(f"Delta uy across H (nominal)  ~ {dUy_nom:.6e} m")

# ----------------------------
# Solve with Dirichlet elimination
# ----------------------------
def solve_dirichlet_elimination(
    K: sp.csr_matrix,
    f: np.ndarray,
    fixed_dofs: np.ndarray,
    fixed_vals: np.ndarray
) -> np.ndarray:
    ndof = K.shape[0]
    all_dofs = np.arange(ndof, dtype=int)
    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    fixed_vals = np.asarray(fixed_vals, dtype=float)
    if fixed_dofs.size != fixed_vals.size:
        raise ValueError("fixed_dofs and fixed_vals must have same length.")

    free = np.setdiff1d(all_dofs, fixed_dofs)

    f_mod = f.copy()
    f_mod[free] -= K[free][:, fixed_dofs] @ fixed_vals

    Kff = K[free][:, free].tocsr()
    ff = f_mod[free]

    logging.info("Solving Kff * uf = ff ...")
    uf = spla.spsolve(Kff, ff)

    u = np.zeros(ndof, dtype=float)
    u[free] = uf
    u[fixed_dofs] = fixed_vals
    return u


# ----------------------------
# Output
# ----------------------------
def write_vtk(cfg: SolverConfig, pts: np.ndarray, quad: np.ndarray, u: np.ndarray):
    cfg.out_vtk.parent.mkdir(parents=True, exist_ok=True)

    pts3 = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(pts.shape[0])])
    U3 = np.column_stack([u[0::2], u[1::2], np.zeros(pts.shape[0])])

    # --- compute cell stresses + von Mises ---
    sig_cell, vm_cell = compute_cell_stress_vonmises(cfg, pts, quad, u)

    # Make sure shapes are correct
    sig_cell = np.asarray(sig_cell, dtype=float)          # (ne,3)
    vm_cell  = np.asarray(vm_cell, dtype=float).reshape(-1)  # (ne,)

    if sig_cell.shape[0] != quad.shape[0] or vm_cell.shape[0] != quad.shape[0]:
        raise RuntimeError(
            f"Cell-data length mismatch: ne={quad.shape[0]}, "
            f"sigma={sig_cell.shape}, vm={vm_cell.shape}"
        )

    mesh = meshio.Mesh(
        points=pts3,
        cells=[("quad", quad)],
        point_data={"U": U3},
        # meshio expects a LIST with one entry per cell block
        cell_data={
            "sigma": [sig_cell],            # (sxx, syy, txy) per element
            "von_mises": [vm_cell],         # scalar per element
            "sigma_xx": [sig_cell[:, 0]],
            "sigma_yy": [sig_cell[:, 1]],
            "tau_xy":   [sig_cell[:, 2]],
        },
    )

    # Write as VTU (most reliable in ParaView)
    mesh.write(str(cfg.out_vtk))
    logging.info(f"Wrote results: {cfg.out_vtk} (with sigma, von_mises cell data)")

def save_npz_results(npz_path: Path, pts: np.ndarray, quad: np.ndarray, u: np.ndarray):
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        pts=pts,
        conn=quad,
        u=u
    )
    logging.info(f"Wrote NPZ: {npz_path}")
    
def save_metadata(cfg: SolverConfig, J: float, KI: float):
    meta = {
        "run_name": cfg.run_name,
        "W": cfg.W,
        "H": cfg.H,
        "a": cfg.a,
        "crack_gap": cfg.crack_gap,
        "E": cfg.E,
        "nu": cfg.nu,
        "plane_stress": cfg.plane_stress,
        "thickness": cfg.thickness,
        "traction_top": list(cfg.traction_top),
        "traction_bottom": list(cfg.traction_bottom),
        "J": J,
        "KI": KI,
    }
    with open(cfg.meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logging.info(f"Wrote metadata: {cfg.meta_path}")

# ----------------------------
# Main
# ----------------------------
def main():
    setup_logging()
    
    cfg = SolverConfig()

    cfg.run_name = "meshrun_0.055mm"
    cfg.run_dir = Path("Data/New Data") / cfg.run_name

    cfg.msh_path = cfg.run_dir / "plate_edge_crack_q4.msh"
    cfg.out_vtk = cfg.run_dir / "solution_q4.vtk"
    cfg.out_npz = cfg.run_dir / "fields.npz"
    cfg.meta_path = cfg.run_dir / "metadata.json"
    runtime_cfg_path = cfg.run_dir / "runtime_config.json"

    runtime_cfg = load_runtime_config(runtime_cfg_path)
    geom_cfg = runtime_cfg.get("geometry", {})
    mat_cfg = runtime_cfg.get("material", {})
    if geom_cfg:
        cfg.W = float(geom_cfg.get("W", cfg.W))
        cfg.H = float(geom_cfg.get("H", cfg.H))
        cfg.a = float(geom_cfg.get("a", cfg.a))
        cfg.crack_gap = float(geom_cfg.get("crack_gap", cfg.crack_gap))
    if mat_cfg:
        cfg.E = float(mat_cfg.get("E", cfg.E))
        cfg.nu = float(mat_cfg.get("nu", cfg.nu))
        cfg.plane_stress = bool(mat_cfg.get("plane_stress", cfg.plane_stress))
        cfg.thickness = float(mat_cfg.get("thickness", cfg.thickness))

    logging.info(f"Reading mesh: {cfg.msh_path}")
    pts, quad, lines, line_phys, phys_map = read_gmsh_quad_mesh(cfg.msh_path)
    logging.info(f"Nodes: {pts.shape[0]}, Quads: {quad.shape[0]}")
    logging.info(f"Physical groups found: {sorted(set(phys_map.values()))}")

    if lines is None:
        raise RuntimeError("Mesh contains no line elements. Ensure Gmsh wrote boundary lines.")
    if line_phys is None:
        logging.warning("gmsh:physical tags for lines not found; will use coordinate-based boundary detection.")

    # ---- Boundary extraction (prefer physical groups; fallback to coordinates) ----
    use_phys = (
        line_phys is not None
        and ("TOP" in phys_map.values())
        and ("BOTTOM" in phys_map.values())
        and ("RIGHT" in phys_map.values())
    )

    if use_phys:
        top_edges = boundary_edges_by_name(lines, line_phys, phys_map, "TOP")
        bottom_edges = boundary_edges_by_name(lines, line_phys, phys_map, "BOTTOM")
    else:
        tol = 1e-8
        top_edges = boundary_edges_by_coord(lines, pts, "TOP", W=cfg.W, H=cfg.H, tol=tol)
        bottom_edges = boundary_edges_by_coord(lines, pts, "BOTTOM", W=cfg.W, H=cfg.H, tol=tol)

    logging.info(f"TOP edges: {len(top_edges)}, BOTTOM edges: {len(bottom_edges)}")
    logging.info(f"traction_top = {cfg.traction_top}, traction_bottom = {cfg.traction_bottom}")

    if len(top_edges) == 0 or len(bottom_edges) == 0:
        raise RuntimeError(
            "Boundary edge detection returned an empty set. "
            "If using coordinate fallback, increase tol or confirm W/H."
        )

    # Assemble stiffness
    K = assemble_global_K(cfg, pts, quad)

    # Traction RHS on TOP and BOTTOM
    f_top = assemble_uniform_traction_rhs(cfg, pts, top_edges, tx=cfg.traction_top[0], ty=cfg.traction_top[1])
    f_bot = assemble_uniform_traction_rhs(cfg, pts, bottom_edges, tx=cfg.traction_bottom[0], ty=cfg.traction_bottom[1])
    f = f_top + f_bot

    logging.info(f"RHS norm = {np.linalg.norm(f):.6e}")

    bottom_nodes = np.unique(bottom_edges.reshape(-1))

    # 1) Fix uy = 0 on entire bottom edge
    fixed_dofs = list((2 * bottom_nodes + 1).astype(int))
    fixed_vals = list(np.zeros_like(bottom_nodes, dtype=float))

    # 2) Fix ux = 0 at ONE bottom node to prevent rigid-body motion in x
    x_bottom = pts[bottom_nodes, 0]
    x_center = 0.5 * (np.min(x_bottom) + np.max(x_bottom))
    anchor = int(bottom_nodes[np.argmin(np.abs(x_bottom - x_center))])  # bottom-center node

    fixed_dofs.append(2 * anchor)   # ux of one node
    fixed_vals.append(0.0)

    pairs = sorted(zip(fixed_dofs, fixed_vals), key=lambda t: t[0])
    fixed_dofs = np.array([p[0] for p in pairs], dtype=int)
    fixed_vals = np.array([p[1] for p in pairs], dtype=float)

    # Solve
    u = solve_dirichlet_elimination(K, f, fixed_dofs, fixed_vals)
    
    # Validate
    validate_solution(cfg, pts, u, K, f, top_edges, bottom_edges)

    # Output
    write_vtk(cfg, pts, quad, u)
    
    save_npz_results(cfg.out_npz, pts, quad, u)

    umax = float(np.max(np.sqrt(u[0::2] ** 2 + u[1::2] ** 2)))
    logging.info(f"Max |U| = {umax:.6e} m")
    
    from src.J_Integral import compute_J_domain_q4, sweep_J_rout
    from src.save_runs import save_run_results

    tip = np.array([cfg.a, 0.0])

    # A good starting point: exclude within ~5*gap from the crack line
    crack_face_excl = 5*cfg.crack_gap

    J, KI = compute_J_domain_q4(
        pts=pts, conn=quad, U=u, tip=tip,
        E=cfg.E, nu=cfg.nu, plane_stress=cfg.plane_stress,
        r_in=0.015, r_out=0.035,
        crack_dir=np.array([1.0, 0.0]),
        crack_start=np.array([0.0, 0.0]),
        crack_end=np.array([cfg.a, 0.0]),
        exclude_crack_faces=True,
        crack_face_exclusion=crack_face_excl,
        log=True
    )
    
    save_metadata(cfg, J, KI)
    update_runtime_config(
        runtime_cfg_path,
        stage="solver",
        updates={
            "run": {"name": cfg.run_name, "run_dir": cfg.run_dir},
            "geometry": geometry_payload(
                geometry_type="plate_edge_crack",
                W=cfg.W,
                H=cfg.H,
                a=cfg.a,
                crack_gap=cfg.crack_gap,
            ),
            "material": material_payload(
                E=cfg.E,
                nu=cfg.nu,
                plane_stress=cfg.plane_stress,
                thickness=cfg.thickness,
            ),
            "stage": {
                "mesh_path": cfg.msh_path,
                "solution_vtk": cfg.out_vtk,
                "fields_npz": cfg.out_npz,
                "metadata_json": cfg.meta_path,
                "resolved": {
                    "J": float(J),
                    "KI": float(KI),
                    "tip": [cfg.a, 0.0],
                    "crack_start": [0.0, 0.0],
                    "crack_dir": [1.0, 0.0],
                },
            },
        },
    )

    res = sweep_J_rout(
        pts=pts, conn=quad, U=u, tip=tip,
        E=cfg.E, nu=cfg.nu, plane_stress=cfg.plane_stress,
        r_in=0.01,
        r_out_list=[0.02, 0.025, 0.03, 0.035],
        crack_dir=np.array([1.0, 0.0]),
        crack_start=np.array([0.0, 0.0]),
        crack_end=np.array([cfg.a, 0.0]),
        exclude_crack_faces=True,
        crack_face_exclusion=crack_face_excl,   # <-- SAME VALUE
        log_each=True
    )
    
    save_run_results(
        folder=str(cfg.run_dir),
        filename=cfg.run_name,
        J=J,
        KI=KI,
        cfg=cfg
    )

if __name__ == "__main__":
    main()
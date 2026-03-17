#!/usr/bin/env python3

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import meshio

from datetime import datetime
import json

# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    base_out_dir: Path = Path("Data/New Data")
    run_name: Optional[str] = None
    out_dir: Optional[Path] = None

    # Geometry (m)
    W: float = 0.200
    H: float = 0.100
    a: float = 0.055
    crack_gap: float = 5e-5

    # Mesh sizing
    lc_global: float = 0.006
    lc_tip: float = 0.001
    tip_refine_r: float = 0.010

    # Material
    E: float = 73.1e9
    nu: float = 0.33
    plane_stress: bool = True
    thickness: float = 0.005

    # Loading/BC
    traction: float = 50e6  # Pa


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def make_run_name(prefix: str = "run") -> str:
    return f"{prefix}_{Config.a}mm"

# ----------------------------
# Gmsh: quad mesh with recombination
# ----------------------------
def build_mesh_gmsh_quads(cfg: Config, msh_path: Path):
    import gmsh

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Building QUAD mesh with Gmsh...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("plate_edge_crack_q4")

    W, H, a = cfg.W, cfg.H, cfg.a
    g = cfg.crack_gap
    lcG, lcT, rT = cfg.lc_global, cfg.lc_tip, cfg.tip_refine_r

    occ = gmsh.model.occ

    # Plate
    plate = occ.addRectangle(0.0, -H / 2, 0.0, W, H)

    # Thin slit as a small rectangular cut
    y_top = +g / 2
    y_bot = -g / 2
    p1 = occ.addPoint(0.0, y_top, 0.0, lcT)
    p2 = occ.addPoint(a, y_top, 0.0, lcT)
    p3 = occ.addPoint(a, y_bot, 0.0, lcT)
    p4 = occ.addPoint(0.0, y_bot, 0.0, lcT)

    l_top = occ.addLine(p1, p2)
    l_tip = occ.addLine(p2, p3)
    l_bot = occ.addLine(p4, p3)
    l_mth = occ.addLine(p4, p1)

    slit_loop = occ.addCurveLoop([l_top, l_tip, -l_bot, l_mth])
    slit_surf = occ.addPlaneSurface([slit_loop])

    # Cut slit from plate
    plate_cut, _ = occ.cut([(2, plate)], [(2, slit_surf)], removeObject=True, removeTool=True)
    if not plate_cut:
        raise RuntimeError("Gmsh boolean cut failed.")
    plate_surf = plate_cut[0][1]

    occ.synchronize()

    # ----------------------------
    # Mesh size fields
    # ----------------------------
    # Goal: enforce UNIFORM small elements inside a disk around the crack tip
    # that fully contains the J-integral annulus (e.g., r_out up to 0.035 m).
    x_tip, y_tip = a, 0.0
    r_uniform = 0.04      # must be >= your largest r_out (recommend 0.04 for r_out<=0.035)
    lcU = lcT              # uniform size inside disk (use lc_tip)
    lcG = lcG              # global outside

    # Field A: uniform refinement ball around crack tip
    gmsh.model.mesh.field.add("Ball", 10)
    gmsh.model.mesh.field.setNumber(10, "XCenter", x_tip)
    gmsh.model.mesh.field.setNumber(10, "YCenter", y_tip)
    gmsh.model.mesh.field.setNumber(10, "ZCenter", 0.0)
    gmsh.model.mesh.field.setNumber(10, "Radius", r_uniform)
    gmsh.model.mesh.field.setNumber(10, "VIn", lcU)     # constant inside ball
    gmsh.model.mesh.field.setNumber(10, "VOut", lcG)    # outside ball

    # Field B: optional extra refinement near crack edges (keeps faces/tip clean)
    gmsh.model.mesh.field.add("Distance", 11)
    gmsh.model.mesh.field.setNumbers(11, "EdgesList", [l_top, l_tip, l_bot])
    gmsh.model.mesh.field.setNumber(11, "Sampling", 80)

    gmsh.model.mesh.field.add("Threshold", 12)
    gmsh.model.mesh.field.setNumber(12, "InField", 11)
    gmsh.model.mesh.field.setNumber(12, "SizeMin", lcU)
    gmsh.model.mesh.field.setNumber(12, "SizeMax", lcU)     # keep it UNIFORM (important)
    gmsh.model.mesh.field.setNumber(12, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(12, "DistMax", r_uniform)

    # Field C: take the minimum of the two (most refined wins)
    gmsh.model.mesh.field.add("Min", 13)
    gmsh.model.mesh.field.setNumbers(13, "FieldsList", [10, 12])

    gmsh.model.mesh.field.setAsBackgroundMesh(13)

    # ----------------------------
    # Physical groups: robust boundary tagging (FIXED)
    # ----------------------------
    bnd = gmsh.model.getBoundary([(2, plate_surf)], oriented=False, recursive=False)
    curves = [c[1] for c in bnd]

    # Use midpoint-based classification with a robust tolerance (NOT 1e-9)
    tol = 1e-6 * max(cfg.W, cfg.H)

    left, right, top, bottom = [], [], [], []
    for ct in curves:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, ct)
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)

        if abs(xmid - 0.0) < tol:
            left.append(ct)
        elif abs(xmid - W) < tol:
            right.append(ct)
        elif abs(ymid - (H / 2.0)) < tol:
            top.append(ct)
        elif abs(ymid - (-H / 2.0)) < tol:
            bottom.append(ct)

    pg_dom = gmsh.model.addPhysicalGroup(2, [plate_surf])
    gmsh.model.setPhysicalName(2, pg_dom, "DOMAIN")

    if left:
        pg = gmsh.model.addPhysicalGroup(1, left)
        gmsh.model.setPhysicalName(1, pg, "LEFT")
    if right:
        pg = gmsh.model.addPhysicalGroup(1, right)
        gmsh.model.setPhysicalName(1, pg, "RIGHT")
    if top:
        pg = gmsh.model.addPhysicalGroup(1, top)
        gmsh.model.setPhysicalName(1, pg, "TOP")
    if bottom:
        pg = gmsh.model.addPhysicalGroup(1, bottom)
        gmsh.model.setPhysicalName(1, pg, "BOTTOM")

    # ---- QUAD meshing settings ----
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.setRecombine(2, plate_surf)

    try:
        gmsh.option.setNumber("Mesh.Algorithm", 8)
    except Exception:
        pass

    # IMPORTANT: keep mesh linear (your solver is Q4, not Q8)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.model.mesh.generate(2)

    # IMPORTANT: write a meshio-friendly format
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 0)
    # Do NOT set Mesh.SaveAll=1 (it often breaks meshio cell_data consistency)

    gmsh.write(str(msh_path))
    gmsh.finalize()
    logging.info(f"Saved mesh: {msh_path}")


# ----------------------------
# Mesh reading helpers
# ----------------------------
def read_msh_quads(msh_path: Path):
    m = meshio.read(str(msh_path))
    pts = np.asarray(m.points[:, :2], dtype=float)

    quad = None
    for block in m.cells:
        if block.type in ("quad", "quad4", "quadrilateral"):
            quad = np.asarray(block.data, dtype=int)
            break
    if quad is None:
        raise ValueError("No quad elements found. (Check recombination settings.)")

    # boundary lines + physical tags
    lines, line_phys = None, None
    for i, block in enumerate(m.cells):
        if block.type == "line":
            lines = np.asarray(block.data, dtype=int)
            if m.cell_data and "gmsh:physical" in m.cell_data:
                line_phys = np.asarray(m.cell_data["gmsh:physical"][i], dtype=int)
            break

    phys_map = {}
    if hasattr(m, "field_data") and m.field_data:
        for name, (pid, dim) in m.field_data.items():
            phys_map[int(pid)] = name

    return pts, quad, lines, line_phys, phys_map


def get_boundary_edges(lines, line_phys, phys_map, name: str) -> np.ndarray:
    if lines is None or line_phys is None:
        raise ValueError(
            "No line elements or physical tags found. "
            "Make sure you created 1D physical groups (LEFT/RIGHT/...) in Gmsh and wrote them to .msh."
        )
    pid = None
    for k, v in phys_map.items():
        if v == name:
            pid = k
            break
    if pid is None:
        raise ValueError(f"Physical group '{name}' not found. Available: {sorted(set(phys_map.values()))}")
    return lines[line_phys == pid]


# ----------------------------
# Q4 FEM
# ----------------------------
def D_matrix(E: float, nu: float, plane_stress: bool) -> np.ndarray:
    if plane_stress:
        c = E / (1.0 - nu**2)
        return c * np.array([[1.0, nu, 0.0],
                             [nu, 1.0, 0.0],
                             [0.0, 0.0, (1.0 - nu)/2.0]], dtype=float)
    c = E / ((1.0 + nu) * (1.0 - 2.0*nu))
    return c * np.array([[1.0 - nu, nu, 0.0],
                         [nu, 1.0 - nu, 0.0],
                         [0.0, 0.0, (1.0 - 2.0*nu)/2.0]], dtype=float)


def q4_shape(xi: float, eta: float):
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
        [-0.25*(1+eta),  0.25*(1-xi)]
    ], dtype=float)
    return N, dN_dxi


def q4_B_matrix(xe: np.ndarray, xi: float, eta: float) -> Tuple[np.ndarray, float]:
    _, dN_dxi = q4_shape(xi, eta)          # (4,2)
    J = xe.T @ dN_dxi                      # (2,2)
    detJ = float(np.linalg.det(J))
    if detJ <= 0:
        raise ValueError(f"Non-positive detJ={detJ}")
    invJ = np.linalg.inv(J)
    dN_dx = dN_dxi @ invJ.T                # (4,2)

    B = np.zeros((3, 8), dtype=float)
    for a in range(4):
        dNdx, dNdy = dN_dx[a, 0], dN_dx[a, 1]
        B[0, 2*a]     = dNdx
        B[1, 2*a+1]   = dNdy
        B[2, 2*a]     = dNdy
        B[2, 2*a+1]   = dNdx
    return B, detJ


def assemble_K_q4(cfg: Config, pts: np.ndarray, quad: np.ndarray) -> sp.csr_matrix:
    n = pts.shape[0]
    ndof = 2*n
    D = D_matrix(cfg.E, cfg.nu, cfg.plane_stress)

    g = 1.0/np.sqrt(3.0)
    gps = [(-g, -g), (g, -g), (g, g), (-g, g)]
    w = 1.0

    rows, cols, vals = [], [], []
    t = cfg.thickness

    logging.info("Assembling Q4 stiffness matrix...")
    for e in range(quad.shape[0]):
        nodes = quad[e]
        xe = pts[nodes, :]  # (4,2)

        Ke = np.zeros((8, 8), dtype=float)
        for (xi, eta) in gps:
            B, detJ = q4_B_matrix(xe, xi, eta)
            Ke += (B.T @ D @ B) * (detJ * w * w * t)

        dofs = np.array([2*nodes[0], 2*nodes[0]+1,
                         2*nodes[1], 2*nodes[1]+1,
                         2*nodes[2], 2*nodes[2]+1,
                         2*nodes[3], 2*nodes[3]+1], dtype=int)

        for i_local in range(8):
            for j_local in range(8):
                rows.append(dofs[i_local])
                cols.append(dofs[j_local])
                vals.append(Ke[i_local, j_local])

    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    logging.info(f"K assembled: shape={K.shape}, nnz={K.nnz}")
    return K


def assemble_traction_rhs(cfg: Config, pts: np.ndarray, edges: np.ndarray, tx: float, ty: float) -> np.ndarray:
    n = pts.shape[0]
    f = np.zeros(2*n, dtype=float)
    tvec = np.array([tx, ty], dtype=float)

    for (n1, n2) in edges:
        x1 = pts[n1]
        x2 = pts[n2]
        L = float(np.linalg.norm(x2 - x1))
        fe_node = cfg.thickness * L * 0.5 * tvec
        f[2*n1:2*n1+2] += fe_node
        f[2*n2:2*n2+2] += fe_node
    return f


def apply_dirichlet(K: sp.csr_matrix, f: np.ndarray, fixed_dofs: np.ndarray, fixed_vals: np.ndarray):
    ndof = K.shape[0]
    all_dofs = np.arange(ndof)
    free = np.setdiff1d(all_dofs, fixed_dofs)

    f_mod = f.copy()
    f_mod[free] -= K[free][:, fixed_dofs] @ fixed_vals

    Kff = K[free][:, free].tocsr()
    ff = f_mod[free]
    return Kff, ff, free


def solve_system(K: sp.csr_matrix, f: np.ndarray, fixed_dofs: np.ndarray, fixed_vals: np.ndarray) -> np.ndarray:
    Kff, ff, free = apply_dirichlet(K, f, fixed_dofs, fixed_vals)
    logging.info("Solving...")
    uf = spla.spsolve(Kff, ff)
    u = np.zeros(K.shape[0], dtype=float)
    u[free] = uf
    u[fixed_dofs] = fixed_vals
    return u


def write_vtk(out_path: Path, pts: np.ndarray, quad: np.ndarray, u: np.ndarray):
    disp = np.column_stack([u[0::2], u[1::2], np.zeros(pts.shape[0])])
    pts3 = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(pts.shape[0])])
    mesh = meshio.Mesh(points=pts3, cells=[("quad", quad)], point_data={"U": disp})
    mesh.write(str(out_path))
    logging.info(f"Wrote: {out_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    setup_logging()
    cfg = Config()
    if cfg.run_name is None:
        cfg.run_name = make_run_name("meshrun")

    cfg.out_dir = cfg.base_out_dir / cfg.run_name
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    msh_path = cfg.out_dir / "plate_edge_crack_q4.msh"

    build_mesh_gmsh_quads(cfg, msh_path)
    pts, quad, lines, line_phys, phys_map = read_msh_quads(msh_path)

    logging.info(f"Mesh: nodes={pts.shape[0]}, quads={quad.shape[0]}")
    logging.info(f"Physical groups: {phys_map}")

    # Sanity check: make sure boundary physicals exist
    if "LEFT" not in phys_map.values() or "RIGHT" not in phys_map.values():
        raise RuntimeError(
            "Mesh is missing LEFT/RIGHT physical groups. "
            "Check boundary tagging tolerance or print curves' bounding boxes."
        )

    K = assemble_K_q4(cfg, pts, quad)

    right_edges = get_boundary_edges(lines, line_phys, phys_map, "RIGHT")
    f = assemble_traction_rhs(cfg, pts, right_edges, tx=cfg.traction, ty=0.0)

    left_edges = get_boundary_edges(lines, line_phys, phys_map, "LEFT")
    left_nodes = np.unique(left_edges.reshape(-1))
    fixed_dofs = np.sort(np.concatenate([2*left_nodes, 2*left_nodes + 1]))
    fixed_vals = np.zeros_like(fixed_dofs, dtype=float)

    u = solve_system(K, f, fixed_dofs, fixed_vals)

    umax = np.max(np.sqrt(u[0::2]**2 + u[1::2]**2))
    logging.info(f"Max |U| = {umax:.6e} m")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Python FEM pipeline (Q4 quads) + mesh plotting

What this script does:
1) Build CAD + quad mesh in Gmsh: plate with central hole + slit crack from hole edge
2) Solve 2D linear elasticity (plane stress) using Q4 bilinear quads (2x2 Gauss)
3) Apply traction on RIGHT edge + Dirichlet fix on LEFT edge
4) Export:
   - mesh (.msh)
   - mesh plot (.png)
   - nodal displacement CSV (node_id, x, y, ux, uy) for DCM post-processing

Notes:
- Units: SI (m, Pa). Traction is in Pa. Thickness in m.
- Crack is modeled geometrically as a thin slit (two faces) created by subtracting a thin rectangle.
- Mesh uses Gmsh recombination to produce quads. If any triangles remain, we error (by design).
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import yaml
import meshio
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


# -----------------------------
# Config containers
# -----------------------------
@dataclass
class GeometryCfg:
    W: float
    H: float
    R: float
    a: float
    crack_opening: float
    tip_refine_radius: float


@dataclass
class MeshCfg:
    lc_global: float
    lc_tip: float
    order: int = 1


@dataclass
class MaterialCfg:
    E: float
    nu: float
    plane_stress: bool
    thickness: float


@dataclass
class LoadingCfg:
    type: str
    traction_pa: float
    fix_left: bool
    fix_left_y: bool


@dataclass
class PostCfg:
    export_dir: str
    export_prefix: str
    crack_line_y: float
    crack_band: float
    x_tip: Optional[float] = None


@dataclass
class Config:
    geometry: GeometryCfg
    mesh: MeshCfg
    material: MaterialCfg
    loading: LoadingCfg
    post: PostCfg


# -----------------------------
# Utilities
# -----------------------------
def load_config(path: Path) -> Config:
    d = yaml.safe_load(path.read_text())
    return Config(
        geometry=GeometryCfg(**d["geometry"]),
        mesh=MeshCfg(**d["mesh"]),
        material=MaterialCfg(**d["material"]),
        loading=LoadingCfg(**d["loading"]),
        post=PostCfg(**d["post"]),
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Meshing with Gmsh (QUADS)
# -----------------------------
def build_mesh_gmsh(cfg: Config, out_msh: Path) -> float:
    """
    Build CAD + quad mesh. Returns computed x_tip (m).
    Ensures boundary line elements are written to the .msh (Mesh.SaveAll=1).
    """
    import gmsh

    logging.info("Building CAD + quad mesh with Gmsh...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("plate_hole_crack_q4")

    W = cfg.geometry.W
    H = cfg.geometry.H
    R = cfg.geometry.R
    a = cfg.geometry.a
    gap = cfg.geometry.crack_opening
    lcG = cfg.mesh.lc_global
    lcTip = cfg.mesh.lc_tip
    tipR = cfg.geometry.tip_refine_radius

    # Hole centered
    cx = W / 2.0
    cy = 0.0

    # Crack from hole +x edge to the right
    x0 = cx + R
    x_tip = x0 + a

    # Slit rectangle (thin cut)
    yL = cy - gap / 2.0

    # Outer plate
    rect = gmsh.model.occ.addRectangle(0.0, -H / 2.0, 0.0, W, H)
    # Hole
    disk = gmsh.model.occ.addDisk(cx, cy, 0.0, R, R)
    # Slit cut
    slit = gmsh.model.occ.addRectangle(x0, yL, 0.0, a, gap)

    # plate \ (hole U slit)
    cut, _ = gmsh.model.occ.cut([(2, rect)], [(2, disk), (2, slit)], removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()

    if not cut:
        gmsh.finalize()
        raise RuntimeError("Gmsh OCC cut() returned empty result; geometry creation failed.")

    # Domain physical group
    surf_tag = cut[0][1]
    gmsh.model.addPhysicalGroup(2, [surf_tag], tag=1)
    gmsh.model.setPhysicalName(2, 1, "DOMAIN")

    # Boundary physical groups via bounding boxes
    # Boundary physical groups via bounding boxes
    eps = 1e-6
    left_curves = gmsh.model.getEntitiesInBoundingBox(-eps, -H / 2 - eps, -eps, eps, H / 2 + eps, eps, 1)
    right_curves = gmsh.model.getEntitiesInBoundingBox(W - eps, -H / 2 - eps, -eps, W + eps, H / 2 + eps, eps, 1)

    top_curves = gmsh.model.getEntitiesInBoundingBox(-eps,  H / 2 - eps, -eps, W + eps,  H / 2 + eps, eps, 1)
    bot_curves = gmsh.model.getEntitiesInBoundingBox(-eps, -H / 2 - eps, -eps, W + eps, -H / 2 + eps, eps, 1)

    left_ids = [c[1] for c in left_curves]
    right_ids = [c[1] for c in right_curves]
    top_ids = [c[1] for c in top_curves]
    bot_ids = [c[1] for c in bot_curves]

    if left_ids:
        gmsh.model.addPhysicalGroup(1, left_ids, tag=11)
        gmsh.model.setPhysicalName(1, 11, "LEFT")
    else:
        logging.warning("No LEFT boundary curves found by bounding box.")

    if right_ids:
        gmsh.model.addPhysicalGroup(1, right_ids, tag=12)
        gmsh.model.setPhysicalName(1, 12, "RIGHT")
    else:
        logging.warning("No RIGHT boundary curves found by bounding box.")

    if top_ids:
        gmsh.model.addPhysicalGroup(1, top_ids, tag=13)
        gmsh.model.setPhysicalName(1, 13, "TOP")
    else:
        logging.warning("No TOP boundary curves found by bounding box.")

    if bot_ids:
        gmsh.model.addPhysicalGroup(1, bot_ids, tag=14)
        gmsh.model.setPhysicalName(1, 14, "BOTTOM")
    else:
        logging.warning("No BOTTOM boundary curves found by bounding box.")

    # (Optional) all boundary curves group for debugging/BC alternatives
    all_curves = gmsh.model.getBoundary([(2, surf_tag)], oriented=False, recursive=False)
    all_curve_ids = sorted({c[1] for c in all_curves if c[0] == 1})
    if all_curve_ids:
        gmsh.model.addPhysicalGroup(1, all_curve_ids, tag=10)
        gmsh.model.setPhysicalName(1, 10, "BOUNDARY_ALL")

    # Tip refinement (distance field to a point at crack tip)
    tip_pt = gmsh.model.occ.addPoint(x_tip, cy, 0.0, lcTip)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "PointsList", [tip_pt])

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", lcTip)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", lcG)
    gmsh.model.mesh.field.setNumber(2, "DistMin", tipR * 0.3)
    gmsh.model.mesh.field.setNumber(2, "DistMax", tipR)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    # Quad recombination settings
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8)                  # Frontal-Delaunay for quads
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lcTip)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lcG)

    # Ensure we write ALL elements (including 1D boundary line elements)
    gmsh.option.setNumber("Mesh.SaveAll", 0)

    # Force recombine on this surface
    gmsh.model.mesh.setRecombine(2, surf_tag, angle=45)

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # If user requested higher order, set it after generation.
    # Force Q4 for this pipeline (higher order caused inverted elements with recombination)
    gmsh.model.mesh.setOrder(1)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(out_msh))
    gmsh.finalize()

    logging.info("Wrote mesh: %s", out_msh.resolve())
    logging.info("Computed crack tip x_tip = %.6f m", x_tip)
    return x_tip


# -----------------------------
# Q4 plane stress FE
# -----------------------------
def D_plane_stress(E: float, nu: float) -> np.ndarray:
    E = float(E)
    nu = float(nu)
    c = E / (1.0 - nu**2)
    return c * np.array(
        [[1.0, nu, 0.0],
         [nu, 1.0, 0.0],
         [0.0, 0.0, (1.0 - nu) / 2.0]],
        dtype=float
    )


def q4_shape_derivs(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    dN_dxi = 0.25 * np.array([-(1 - eta),
                               (1 - eta),
                               (1 + eta),
                              -(1 + eta)], dtype=float)
    dN_deta = 0.25 * np.array([-(1 - xi),
                               -(1 + xi),
                                (1 + xi),
                                (1 - xi)], dtype=float)
    return dN_dxi, dN_deta


def reorder_quad_ccw(xy: np.ndarray) -> np.ndarray:
    c = xy.mean(axis=0)
    ang = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
    order = np.argsort(ang)
    return order


def q4_B_matrix(xy: np.ndarray, xi: float, eta: float) -> Tuple[np.ndarray, float]:
    dN_dxi, dN_deta = q4_shape_derivs(xi, eta)

    J = np.zeros((2, 2), dtype=float)
    J[0, 0] = float(np.dot(dN_dxi,  xy[:, 0]))
    J[0, 1] = float(np.dot(dN_deta, xy[:, 0]))
    J[1, 0] = float(np.dot(dN_dxi,  xy[:, 1]))
    J[1, 1] = float(np.dot(dN_deta, xy[:, 1]))

    detJ = float(np.linalg.det(J))
    if detJ <= 0:
        raise ValueError("Non-positive detJ (inverted quad).")

    invJ = np.linalg.inv(J)
    grads = invJ @ np.vstack((dN_dxi, dN_deta))  # (2,4)
    dN_dx = grads[0, :]
    dN_dy = grads[1, :]

    B = np.zeros((3, 8), dtype=float)
    for i in range(4):
        B[0, 2*i]     = dN_dx[i]
        B[1, 2*i + 1] = dN_dy[i]
        B[2, 2*i]     = dN_dy[i]
        B[2, 2*i + 1] = dN_dx[i]

    return B, detJ


def q4_element_stiffness(xy: np.ndarray, D: np.ndarray, t: float) -> np.ndarray:
    Ke = np.zeros((8, 8), dtype=float)
    gp = 1.0 / math.sqrt(3.0)
    gauss = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    for (xi, eta) in gauss:
        B, detJ = q4_B_matrix(xy, xi, eta)
        Ke += (B.T @ D @ B) * (t * detJ)
    return Ke


def assemble_system_q4(points: np.ndarray, quads: np.ndarray, D: np.ndarray, t: float) -> csr_matrix:
    n = points.shape[0]
    ndof = 2 * n
    K = lil_matrix((ndof, ndof), dtype=float)

    for conn in quads:
        xy = points[conn, :]
        order = reorder_quad_ccw(xy)
        conn2 = conn[order]
        xy2 = points[conn2, :]

        Ke = q4_element_stiffness(xy2, D, t)

        dofs = np.array([2*conn2[0], 2*conn2[0]+1,
                         2*conn2[1], 2*conn2[1]+1,
                         2*conn2[2], 2*conn2[2]+1,
                         2*conn2[3], 2*conn2[3]+1], dtype=int)

        for i in range(8):
            for j in range(8):
                K[dofs[i], dofs[j]] += Ke[i, j]

    return K.tocsr()


def build_load_vector_traction(
    points: np.ndarray,
    line_elems: np.ndarray,
    traction_pa: float,
    t: float,
    direction: str,) -> np.ndarray:
    """
    Uniform traction on a set of boundary line elements.

    direction:
      - "RIGHT":  tx = +traction, ty = 0
      - "LEFT":   tx = -traction, ty = 0
      - "TOP":    tx = 0, ty = +traction
      - "BOTTOM": tx = 0, ty = -traction
    """
    n = points.shape[0]
    f = np.zeros(2 * n, dtype=float)

    d = direction.upper()
    if d == "RIGHT":
        tx, ty = traction_pa, 0.0
    elif d == "LEFT":
        tx, ty = -traction_pa, 0.0
    elif d == "TOP":
        tx, ty = 0.0, traction_pa
    elif d == "BOTTOM":
        tx, ty = 0.0, -traction_pa
    else:
        raise ValueError(f"Unknown traction direction: {direction}")

    for (n1, n2) in line_elems:
        x1, y1 = points[n1]
        x2, y2 = points[n2]
        L = math.hypot(x2 - x1, y2 - y1)

        # Consistent nodal load for a 2-node line under uniform traction:
        # fe = ∫ N^T * t * traction ds  -> t*L/2 at each node
        fe = (t * L / 2.0) * np.array([tx, ty, tx, ty], dtype=float)

        f[2 * n1 : 2 * n1 + 2] += fe[0:2]
        f[2 * n2 : 2 * n2 + 2] += fe[2:4]

    return f



def apply_dirichlet(K: csr_matrix, f: np.ndarray, fixed_dofs: np.ndarray, values: Optional[np.ndarray] = None) -> Tuple[csr_matrix, np.ndarray]:
    if values is None:
        values = np.zeros_like(fixed_dofs, dtype=float)

    K = K.tolil()
    for dof, val in zip(fixed_dofs, values):
        K.rows[dof] = [dof]
        K.data[dof] = [1.0]
        f[dof] = val
    return K.tocsr(), f


# -----------------------------
# Mesh extraction helpers
# -----------------------------
def _concat_cellblocks(mesh: meshio.Mesh, types: Tuple[str, ...]) -> Optional[np.ndarray]:
    blocks = []
    for cb in mesh.cells:
        if cb.type in types:
            blocks.append(cb.data)
    if not blocks:
        return None
    return np.vstack(blocks)


def _concat_cell_data(mesh: meshio.Mesh, key: str, types: Tuple[str, ...]) -> Optional[np.ndarray]:
    """
    Concatenate cell_data arrays aligned to blocks in mesh.cells for the given types.
    """
    if not mesh.cell_data or key not in mesh.cell_data:
        return None
    blocks = []
    for i, cb in enumerate(mesh.cells):
        if cb.type in types:
            blocks.append(np.asarray(mesh.cell_data[key][i], dtype=int))
    if not blocks:
        return None
    return np.concatenate(blocks)


def extract_cells(mesh: meshio.Mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      points_xy (n,2),
      quads (nq,4),
      lines (nl,2),
      line_phys (nl,) physical tags (or -1 if unavailable)
    """
    points_xy = mesh.points[:, :2].astype(float)

    quads = _concat_cellblocks(mesh, ("quad", "quad4"))
    tris = _concat_cellblocks(mesh, ("triangle", "triangle3"))
    lines = _concat_cellblocks(mesh, ("line", "line2"))

    if tris is not None and len(tris) > 0:
        raise RuntimeError("Mesh contains triangles. Adjust Gmsh quad recombination settings.")

    if quads is None or len(quads) == 0:
        raise RuntimeError("No quad elements found. Check Gmsh recombination settings.")

    if lines is None or len(lines) == 0:
        raise RuntimeError(
            "No boundary line elements found in mesh. "
            "Ensure Gmsh writes 1D elements (Mesh.SaveAll=1) and/or boundary PhysicalGroups."
        )

    line_phys = _concat_cell_data(mesh, "gmsh:physical", ("line", "line2"))
    if line_phys is None:
        line_phys = -np.ones((lines.shape[0],), dtype=int)

    return points_xy, np.asarray(quads, dtype=int), np.asarray(lines, dtype=int), np.asarray(line_phys, dtype=int)


def select_lines_by_phys(lines: np.ndarray, line_phys: np.ndarray, phys_id: int) -> np.ndarray:
    mask = (line_phys == phys_id)
    return lines[mask, :]


# -----------------------------
# Plotting
# -----------------------------
def plot_mesh(points: np.ndarray, quads: np.ndarray, title: str, out_png: Path, show: bool = False) -> None:
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    for conn in quads:
        xy = points[conn, :]
        order = reorder_quad_ccw(xy)
        xy = xy[order, :]
        poly = np.vstack([xy, xy[0]])
        ax.plot(poly[:, 0], poly[:, 1], linewidth=0.5)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    logging.info("Saved mesh plot: %s", out_png.resolve())
    if show:
        plt.show()
    plt.close()
    
def plot_deformed_mesh(
    points: np.ndarray,
    quads: np.ndarray,
    U: np.ndarray,
    title: str,
    out_png: Path,
    scale: float = 1.0,
    show: bool = False,) -> None:
    """
    Plot deformed Q4 mesh: (x',y') = (x,y) + scale*(ux,uy)
    """
    ux = U[0::2]
    uy = U[1::2]
    pts_def = points.copy()
    pts_def[:, 0] += scale * ux
    pts_def[:, 1] += scale * uy

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    for conn in quads:
        xy = pts_def[conn, :]
        order = reorder_quad_ccw(xy)
        xy = xy[order, :]
        poly = np.vstack([xy, xy[0]])
        ax.plot(poly[:, 0], poly[:, 1], linewidth=0.5)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title + f" (scale={scale:g}x)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    logging.info("Saved deformed mesh plot: %s", out_png.resolve())
    if show:
        plt.show()
    plt.close()



# -----------------------------
# Post-processing export
# -----------------------------
def export_nodal_csv(points: np.ndarray, U: np.ndarray, out_csv: Path, y0: float, band: float) -> None:
    n = points.shape[0]
    ux = U[0::2]
    uy = U[1::2]

    sel = np.abs(points[:, 1] - y0) <= band
    ids = np.arange(n, dtype=int)[sel]
    data = np.column_stack([ids, points[sel, 0], points[sel, 1], ux[sel], uy[sel]])

    header = "node_id,x,y,ux,uy"
    np.savetxt(out_csv, data, delimiter=",", header=header, comments="")
    logging.info("Wrote nodal displacement CSV (%d rows): %s", data.shape[0], out_csv.resolve())


# -----------------------------
# Main pipeline
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="Path to config.yaml")

    # accept BOTH flags to match your existing command usage
    ap.add_argument("--regen-mesh", dest="regen_mesh", action="store_true", help="Regenerate mesh with Gmsh")
    ap.add_argument("--regen-m", dest="regen_mesh", action="store_true", help="Alias for --regen-mesh")

    ap.add_argument("--show-mesh", action="store_true", help="Show mesh plot interactively (also saves PNG)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config(args.config)

    export_dir = (args.config.parent / Path(cfg.post.export_dir)).resolve()
    ensure_dir(export_dir)

    prefix = cfg.post.export_prefix
    msh_path = export_dir / f"{prefix}_mesh.msh"
    mesh_png = export_dir / f"{prefix}_mesh.png"
    disp_csv = export_dir / f"{prefix}_nodes_for_dcm.csv"
    mesh_def_png = export_dir / f"{prefix}_mesh_deformed.png"


    # 1) Mesh
    if args.regen_mesh or (not msh_path.exists()):
        _ = build_mesh_gmsh(cfg, msh_path)

    # 2) Read mesh
    mesh = meshio.read(msh_path)
    points, quads, lines, line_phys = extract_cells(mesh)

    # Plot mesh
    plot_mesh(points, quads, title=f"{prefix}: Q4 mesh", out_png=mesh_png, show=args.show_mesh)

    # 3) Build FE system
    if not cfg.material.plane_stress:
        raise NotImplementedError("This starter pipeline currently implements plane stress only.")

    D = D_plane_stress(cfg.material.E, cfg.material.nu)
    t = cfg.material.thickness

    logging.info("Assembling global stiffness (Q4)...")
    K = assemble_system_q4(points, quads, D, t)

    # 4) Loads: tension in y by traction on TOP (+) and BOTTOM (-)
    traction = float(cfg.loading.traction_pa)
    if cfg.loading.type.lower() != "traction":
        raise NotImplementedError("This pipeline currently supports traction loading only.")

    top_lines = select_lines_by_phys(lines, line_phys, phys_id=13)
    bot_lines = select_lines_by_phys(lines, line_phys, phys_id=14)

    if top_lines.size == 0:
        raise RuntimeError("No TOP boundary lines found (physical tag 13). Check Gmsh physical groups.")
    if bot_lines.size == 0:
        raise RuntimeError("No BOTTOM boundary lines found (physical tag 14). Check Gmsh physical groups.")

    f = np.zeros(2 * points.shape[0], dtype=float)
    f += build_load_vector_traction(points, top_lines, traction, t, direction="TOP")
    f += build_load_vector_traction(points, bot_lines, traction, t, direction="BOTTOM")


    # 5) Dirichlet BCs: fix LEFT and RIGHT (ux=uy=0)
    fixed_dofs: List[int] = []

    left_lines = select_lines_by_phys(lines, line_phys, phys_id=11)
    right_lines = select_lines_by_phys(lines, line_phys, phys_id=12)

    if left_lines.size == 0:
        raise RuntimeError("No LEFT boundary lines found (physical tag 11). Check Gmsh physical groups.")
    if right_lines.size == 0:
        raise RuntimeError("No RIGHT boundary lines found (physical tag 12). Check Gmsh physical groups.")

    left_nodes = np.unique(left_lines.reshape(-1))
    right_nodes = np.unique(right_lines.reshape(-1))
    fixed_nodes = np.unique(np.concatenate([left_nodes, right_nodes]))

    # Fix both components on both sides
    fixed_dofs.extend((2 * fixed_nodes).tolist())       # ux = 0
    fixed_dofs.extend((2 * fixed_nodes + 1).tolist())   # uy = 0

    fixed_dofs_arr = np.unique(np.asarray(fixed_dofs, dtype=int))
    logging.info("Applying Dirichlet BCs on %d dofs", len(fixed_dofs_arr))
    K_bc, f_bc = apply_dirichlet(K, f, fixed_dofs_arr)


    # 6) Solve
    logging.info("Solving Ku=f ...")
    U = spsolve(K_bc, f_bc)
    logging.info("Solve complete. |U|max = %.6e", float(np.max(np.abs(U))))

    # 7) Export nodes near crack line for your DCM
    export_nodal_csv(points, U, disp_csv, y0=cfg.post.crack_line_y, band=cfg.post.crack_band)

        # Choose a reasonable visualization scale automatically
    umax = float(np.max(np.sqrt(U[0::2]**2 + U[1::2]**2)))
    if umax > 0:
        # target ~2% of plate height as visible deformation
        target = 0.02 * cfg.geometry.H
        scale = target / umax
        # clamp so it doesn't get ridiculous
        scale = float(np.clip(scale, 1.0, 5e3))
    else:
        scale = 1.0

    plot_deformed_mesh(
        points, quads, U,
        title=f"{prefix}: deformed mesh",
        out_png=mesh_def_png,
        scale=scale,
        show=args.show_mesh
    )

    logging.info("Done.")
    logging.info("Outputs:")
    logging.info("  Mesh: %s", msh_path)
    logging.info("  Mesh plot: %s", mesh_png)
    logging.info("  DCM nodal CSV: %s", disp_csv)
    logging.info("  Deformed mesh plot: %s", mesh_def_png)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""
stochastic_mesh.py

Geometry-aware quad mesh generator for:
- plate_edge_crack
- plate_hole_edge_crack

Writes:
- mesh_q4.msh
- geometry_metadata.json

Design goals
------------
- Pure linear Q4 mesh only
- Stable for stochastic_fem_solver.py / stochastic_validate_fields.py
- Uses a thin slit cut to represent the crack
- Adds crack-tip-local refinement
- Fails hard if triangles are present

Notes
-----
- Plate domain is [0, W] x [-H/2, H/2]
- For plate_hole_edge_crack:
    hole center defaults to (W/2, 0)
    crack starts at x = cx + hole_radius and grows in +x by length a
"""

from __future__ import annotations

import logging
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

from src.configs.geometry import geometry_payload
from src.configs.material import material_payload
from src.configs.run_io import update_runtime_config


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class MeshConfig:
    base_out_dir: Path = Path("Data/New Data")
    run_name: Optional[str] = None
    out_dir: Optional[Path] = None

    geometry_type: str = "plate_edge_crack"   # or "plate_hole_edge_crack"

    # Plate dimensions
    W: float = 0.2
    H: float = 0.1

    # Crack geometry
    a: float = 0.040
    crack_gap: float = 5e-5

    # Hole geometry (used only for plate_hole_edge_crack)
    hole_radius: float = 0.010
    hole_center: Optional[Tuple[float, float]] = None

    # Mesh sizing
    lc_global: float = 0.006
    lc_tip: float = 0.001
    tip_refine_r: float = 0.005


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def crack_tip_xy(cfg: MeshConfig) -> Tuple[float, float]:
    if cfg.geometry_type == "plate_edge_crack":
        return (cfg.a, 0.0)

    if cfg.geometry_type == "plate_hole_edge_crack":
        cx, cy = cfg.hole_center if cfg.hole_center is not None else (cfg.W / 2.0, 0.0)
        return (cx + cfg.hole_radius + cfg.a, cy)

    raise ValueError(f"Unsupported geometry_type = {cfg.geometry_type!r}")


def crack_start_xy(cfg: MeshConfig) -> Tuple[float, float]:
    if cfg.geometry_type == "plate_edge_crack":
        return (0.0, 0.0)

    if cfg.geometry_type == "plate_hole_edge_crack":
        cx, cy = cfg.hole_center if cfg.hole_center is not None else (cfg.W / 2.0, 0.0)
        return (cx + cfg.hole_radius, cy)

    raise ValueError(f"Unsupported geometry_type = {cfg.geometry_type!r}")


def jsonable_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------
# Physical boundary classification
# ---------------------------------------------------------------------

def classify_outer_boundary_curves(gmsh, plate_surf: int, W: float, H: float):
    """
    Classify exterior boundary curves into LEFT / RIGHT / TOP / BOTTOM.
    Crack slit edges and hole boundaries are excluded automatically because
    they do not lie on the outer plate box boundaries.
    """
    bnd = gmsh.model.getBoundary([(2, plate_surf)], oriented=False, recursive=False)
    curves = [c[1] for c in bnd]

    tol = 1e-6 * max(W, H)
    left, right, top, bottom = [], [], [], []

    for ct in curves:
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, ct)
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)

        if abs(xmid - 0.0) < tol and abs(xmax - xmin) < tol:
            left.append(ct)
        elif abs(xmid - W) < tol and abs(xmax - xmin) < tol:
            right.append(ct)
        elif abs(ymid - (H / 2.0)) < tol and abs(ymax - ymin) < tol:
            top.append(ct)
        elif abs(ymid - (-H / 2.0)) < tol and abs(ymax - ymin) < tol:
            bottom.append(ct)

    return left, right, top, bottom


# ---------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------

def build_mesh_gmsh_quads(cfg: MeshConfig, msh_path: Path) -> None:
    import gmsh

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add(cfg.geometry_type)
        occ = gmsh.model.occ

        W, H = cfg.W, cfg.H
        a = cfg.a
        g = cfg.crack_gap
        lcG = cfg.lc_global
        lcT = cfg.lc_tip

        # -------------------------------------------------------------
        # Base plate: [0, W] x [-H/2, H/2]
        # -------------------------------------------------------------
        plate = occ.addRectangle(0.0, -H / 2.0, 0.0, W, H)

        slit_y_top = +g / 2.0
        slit_y_bot = -g / 2.0
        tool_surfs = []

        # -------------------------------------------------------------
        # Optional hole + crack start location
        # -------------------------------------------------------------
        if cfg.geometry_type == "plate_hole_edge_crack":
            cx, cy = cfg.hole_center if cfg.hole_center is not None else (W / 2.0, 0.0)
            hole = occ.addDisk(cx, cy, 0.0, cfg.hole_radius, cfg.hole_radius)
            tool_surfs.append((2, hole))

            slit_left_x = cx + cfg.hole_radius
            slit_right_x = slit_left_x + a

        elif cfg.geometry_type == "plate_edge_crack":
            slit_left_x = 0.0
            slit_right_x = a

        else:
            raise ValueError(f"Unsupported geometry_type = {cfg.geometry_type!r}")

        # -------------------------------------------------------------
        # Thin slit cut for crack
        # -------------------------------------------------------------
        p1 = occ.addPoint(slit_left_x,  slit_y_top, 0.0, lcT)
        p2 = occ.addPoint(slit_right_x, slit_y_top, 0.0, lcT)
        p3 = occ.addPoint(slit_right_x, slit_y_bot, 0.0, lcT)
        p4 = occ.addPoint(slit_left_x,  slit_y_bot, 0.0, lcT)

        l_top  = occ.addLine(p1, p2)
        l_tip  = occ.addLine(p2, p3)
        l_bot  = occ.addLine(p4, p3)
        l_left = occ.addLine(p4, p1)

        slit_loop = occ.addCurveLoop([l_top, l_tip, -l_bot, l_left])
        slit_surf = occ.addPlaneSurface([slit_loop])
        tool_surfs.append((2, slit_surf))

        # -------------------------------------------------------------
        # Boolean cut
        # -------------------------------------------------------------
        plate_cut, _ = occ.cut([(2, plate)], tool_surfs, removeObject=True, removeTool=True)
        if not plate_cut:
            raise RuntimeError("Boolean cut failed: no plate surface returned.")

        if len(plate_cut) != 1:
            raise RuntimeError(
                f"Expected one final plate surface after cut, got {len(plate_cut)} surfaces."
            )

        plate_surf = plate_cut[0][1]
        occ.synchronize()

        # -------------------------------------------------------------
        # Mesh refinement fields near crack tip
        # -------------------------------------------------------------
        x_tip, y_tip = crack_tip_xy(cfg)
        r_uniform = cfg.tip_refine_r

        # Uniform fine ball around tip
        gmsh.model.mesh.field.add("Ball", 10)
        gmsh.model.mesh.field.setNumber(10, "XCenter", x_tip)
        gmsh.model.mesh.field.setNumber(10, "YCenter", y_tip)
        gmsh.model.mesh.field.setNumber(10, "ZCenter", 0.0)
        gmsh.model.mesh.field.setNumber(10, "Radius", r_uniform)
        gmsh.model.mesh.field.setNumber(10, "VIn", lcT)
        gmsh.model.mesh.field.setNumber(10, "VOut", lcG)

        # Distance-based control along crack edges
        gmsh.model.mesh.field.add("Distance", 11)
        gmsh.model.mesh.field.setNumbers(11, "EdgesList", [l_top, l_tip, l_bot])
        gmsh.model.mesh.field.setNumber(11, "Sampling", 150)

        gmsh.model.mesh.field.add("Threshold", 12)
        gmsh.model.mesh.field.setNumber(12, "InField", 11)
        gmsh.model.mesh.field.setNumber(12, "SizeMin", lcT)
        gmsh.model.mesh.field.setNumber(12, "SizeMax", lcG)
        gmsh.model.mesh.field.setNumber(12, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(12, "DistMax", r_uniform)

        gmsh.model.mesh.field.add("Min", 13)
        gmsh.model.mesh.field.setNumbers(13, "FieldsList", [10, 12])
        gmsh.model.mesh.field.setAsBackgroundMesh(13)

        # -------------------------------------------------------------
        # Physical groups
        # -------------------------------------------------------------
        left, right, top, bottom = classify_outer_boundary_curves(gmsh, plate_surf, W, H)

        pg_dom = gmsh.model.addPhysicalGroup(2, [plate_surf])
        gmsh.model.setPhysicalName(2, pg_dom, "DOMAIN")

        for name, curve_ids in [
            ("LEFT", left),
            ("RIGHT", right),
            ("TOP", top),
            ("BOTTOM", bottom),
        ]:
            if curve_ids:
                pg = gmsh.model.addPhysicalGroup(1, curve_ids)
                gmsh.model.setPhysicalName(1, pg, name)

        # -------------------------------------------------------------
        # Quad mesh controls
        # -------------------------------------------------------------
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        # Match the validated-style quad generation more closely
        try:
            gmsh.option.setNumber("Mesh.Algorithm", 8)
        except Exception:
            pass

        gmsh.model.mesh.setRecombine(2, plate_surf)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(1)

        # -------------------------------------------------------------
        # Hard topology check: must be pure Q4
        # -------------------------------------------------------------
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2, plate_surf)

        type_counts = {}
        n_tri = 0
        n_quad = 0
        n_other = 0

        for et, tags in zip(elem_types, elem_tags):
            name, dim, order, num_nodes, local_coords, _ = gmsh.model.mesh.getElementProperties(et)
            type_counts[name] = len(tags)
            lname = name.lower()

            if "triangle" in lname:
                n_tri += len(tags)
            elif "quadrangle" in lname or "quadrilateral" in lname or "quad" in lname:
                n_quad += len(tags)
            else:
                n_other += len(tags)

        logging.info(f"2D element counts: {type_counts}")

        if n_tri > 0 or n_other > 0:
            gmsh.write(str(msh_path))
            raise RuntimeError(
                "Mesh is not pure Q4. "
                f"Found {n_quad} quads, {n_tri} triangles, {n_other} other 2D elements. "
                "Current stochastic solver assumes all elements are linear quads."
            )

        if n_quad == 0:
            gmsh.write(str(msh_path))
            raise RuntimeError("No quadrilateral elements were generated.")

        # -------------------------------------------------------------
        # Write mesh
        # -------------------------------------------------------------
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.write(str(msh_path))

        # -------------------------------------------------------------
        # Metadata
        # -------------------------------------------------------------
        runtime_cfg_path = cfg.out_dir / "runtime_config.json"
        meta_path = cfg.out_dir / "geometry_metadata.json"

        shared_geometry = geometry_payload(
            geometry_type=cfg.geometry_type,
            W=cfg.W,
            H=cfg.H,
            a=cfg.a,
            crack_gap=cfg.crack_gap,
            hole_radius=cfg.hole_radius if cfg.geometry_type == "plate_hole_edge_crack" else None,
            hole_center=cfg.hole_center if cfg.geometry_type == "plate_hole_edge_crack" else None,
        )

        update_runtime_config(
            runtime_cfg_path,
            stage="mesh",
            updates={
                "run": {"name": cfg.run_name, "base_out_dir": cfg.base_out_dir, "run_dir": cfg.out_dir},
                "geometry": shared_geometry,
                "material": material_payload(E=73.1e9, nu=0.33, plane_stress=True),
                "stage": {
                    "mesh_path": msh_path,
                    "mesh_sizing": {
                        "lc_global": cfg.lc_global,
                        "lc_tip": cfg.lc_tip,
                        "tip_refine_r": cfg.tip_refine_r,
                    },
                    "resolved": {
                        "tip": shared_geometry["tip"],
                        "crack_start": shared_geometry["crack_start"],
                        "crack_dir": shared_geometry["crack_dir"],
                    },
                },
            },
        )

        meta = jsonable_dict(asdict(cfg))
        meta.update(shared_geometry)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        logging.info(f"Saved mesh to: {msh_path}")
        logging.info(f"Saved metadata to: {meta_path}")
        logging.info(f"Saved runtime config to: {runtime_cfg_path}")

    finally:
        gmsh.finalize()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    setup_logging()

    cfg = MeshConfig()

    if cfg.run_name is None:
        cfg.run_name = f"Deterministic_config_{cfg.geometry_type}"

    cfg.out_dir = cfg.base_out_dir / cfg.run_name
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    msh_path = cfg.out_dir / "mesh_q4.msh"
    build_mesh_gmsh_quads(cfg, msh_path)


if __name__ == "__main__":
    main()

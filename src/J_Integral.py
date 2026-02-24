#!/usr/bin/env python3
"""
J_integral.py

Domain-form J-integral for an unstructured Q4 mesh (Gmsh + your FEM solver output).

Implements the standard 2D LEFM domain integral:

    J = ∫_A ( (σ_ij u_{i,1} - W δ_{1j}) q_{,j} ) dA

Patches included:
  1) Smoothstep q-field (C1 continuous) instead of linear ramp.
  2) Exclude Gauss points too close to the crack faces (slit/notch modeling artifact control).

Notes:
  - x1 is the crack growth direction (default: global +x)
  - W = 0.5 * σ:ε is strain energy density (Voigt with engineering shear strain)
  - q is a scalar weight field: q=1 inside r_in, q=0 outside r_out (smoothstep ramp in between)
  - q_,j is the gradient of q

Outputs:
  - J (energy release rate per unit thickness) in N/m
  - KI (Mode I SIF) using:  KI = sqrt(J * E')  (plane stress: E'=E, plane strain: E'=E/(1-ν^2))

Assumptions:
  - Linear elastic material
  - 2D plane stress/plane strain formulation
  - Q4 bilinear elements
  - Displacements U are global DOFs: U[2*i]=ux_i, U[2*i+1]=uy_i
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

import numpy as np


# ----------------------------
# Constitutive
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


def E_prime(E: float, nu: float, plane_stress: bool) -> float:
    # J = K^2 / E'  => K = sqrt(J*E')
    return float(E if plane_stress else E / (1.0 - nu**2))


# ----------------------------
# Q4 shape functions
# ----------------------------
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


def q4_kinematics_at_gp(
    xe: np.ndarray,   # (4,2) nodal coordinates
    ue: np.ndarray,   # (4,2) nodal displacements
    xi: float,
    eta: float
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Returns:
      xgp: (2,) global coordinates at GP
      detJ: float
      dN_dx: (4,2) gradients wrt (x,y)
      grad_u: (2,2) [ [du/dx, du/dy],
                      [dv/dx, dv/dy] ]
    """
    N, dN_dxi = q4_shape(xi, eta)            # (4,), (4,2)
    J = xe.T @ dN_dxi                        # (2,2)
    detJ = float(np.linalg.det(J))
    if detJ <= 0.0:
        raise ValueError(f"Non-positive detJ={detJ}. Element may be inverted.")
    invJ = np.linalg.inv(J)
    dN_dx = dN_dxi @ invJ.T                  # (4,2)

    xgp = N @ xe                              # (2,)
    grad_u = ue.T @ dN_dx                     # (2,2)

    return xgp, detJ, dN_dx, grad_u


# ----------------------------
# Geometry helper: distance from point to line segment
# ----------------------------
def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance from point p to line segment [a,b] in R^2.
    """
    p = np.asarray(p, dtype=float).reshape(2)
    a = np.asarray(a, dtype=float).reshape(2)
    b = np.asarray(b, dtype=float).reshape(2)

    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-30:
        return float(np.linalg.norm(p - a))
    t = float(((p - a) @ ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


# ----------------------------
# q-field (smoothstep radial ramp)
# ----------------------------
def q_field_radial_smoothstep(x: np.ndarray, tip: np.ndarray, r_in: float, r_out: float) -> Tuple[float, np.ndarray]:
    """
    q = 1 for r <= r_in
        smoothstep ramp to 0 for r_in < r < r_out
        0 for r >= r_out

    Smoothstep: s(ξ) = 1 - 3ξ^2 + 2ξ^3,  ξ=(r-r_in)/(r_out-r_in)
      s(0)=1, s(1)=0, s'(0)=s'(1)=0

    Returns:
      q: float
      gradq: (2,)
    """
    dx = x - tip
    r = float(np.linalg.norm(dx))

    if r <= r_in:
        return 1.0, np.zeros(2, dtype=float)
    if r >= r_out:
        return 0.0, np.zeros(2, dtype=float)

    dr = (r_out - r_in)
    if dr <= 0.0:
        raise ValueError("r_out must be > r_in")

    # normalized coordinate in annulus
    xi = (r - r_in) / dr  # in (0,1)
    # smoothstep (C1)
    q = 1.0 - 3.0*xi*xi + 2.0*xi*xi*xi
    dq_dxi = -6.0*xi + 6.0*xi*xi
    dq_dr = dq_dxi / dr

    if r < 1e-14:
        return float(q), np.zeros(2, dtype=float)

    gradq = dq_dr * (dx / r)
    return float(q), gradq


# ----------------------------
# J integral (domain form)
# ----------------------------
def compute_J_domain_q4(
    pts: np.ndarray,          # (N,2)
    conn: np.ndarray,         # (Ne,4) Q4
    U: np.ndarray,            # (2N,)
    tip: np.ndarray,          # (2,)
    E: float,
    nu: float,
    plane_stress: bool,
    r_in: float,
    r_out: float,
    crack_dir: np.ndarray = np.array([1.0, 0.0], dtype=float),
    element_filter_buffer: float = 1.5,
    log: bool = False,
    # ---- new: crack-face exclusion controls ----
    crack_start: Optional[np.ndarray] = None,      # default: [0,0]
    crack_end: Optional[np.ndarray] = None,        # default: tip
    exclude_crack_faces: bool = True,
    crack_face_exclusion: Optional[float] = None,  # distance threshold (m); if None -> 3e-4*r_out
) -> Tuple[float, float]:
    """
    Computes J and KI using the domain integral.

    crack_dir: direction of crack extension (unit vector). Default +x.
    The integral uses derivative u_{i,1} along crack_dir.

    Crack-face exclusion:
      If exclude_crack_faces=True, Gauss points within crack_face_exclusion distance
      from the crack segment [crack_start, crack_end] are skipped.
      This reduces bias when the crack is modeled as a finite-gap slit/notch.

    Returns:
      J  (N/m) per unit thickness
      KI (Pa*sqrt(m))
    """
    pts = np.asarray(pts, dtype=float)
    conn = np.asarray(conn, dtype=int)
    U = np.asarray(U, dtype=float).reshape(-1)
    tip = np.asarray(tip, dtype=float).reshape(2)

    # default crack segment: from (0,0) to tip (works for your edge crack)
    if crack_start is None:
        crack_start = np.array([0.0, 0.0], dtype=float)
    else:
        crack_start = np.asarray(crack_start, dtype=float).reshape(2)

    if crack_end is None:
        crack_end = tip.copy()
    else:
        crack_end = np.asarray(crack_end, dtype=float).reshape(2)

    if crack_face_exclusion is None:
        # conservative default: small fraction of r_out (kept tiny but nonzero)
        crack_face_exclusion = 3e-4 * float(r_out)

    crack_face_exclusion = float(crack_face_exclusion)

    # normalize crack_dir
    crack_dir = np.asarray(crack_dir, dtype=float).reshape(2)
    cdn = float(np.linalg.norm(crack_dir))
    if cdn <= 0:
        raise ValueError("crack_dir must be nonzero.")
    crack_dir = crack_dir / cdn

    D = D_matrix(E, nu, plane_stress)

    # 2x2 Gauss
    g = 1.0 / np.sqrt(3.0)
    gps = [(-g, -g), ( g, -g), ( g,  g), (-g,  g)]
    w = 1.0

    Jval = 0.0

    # quick element filter based on centroid distance (speeds up)
    r_keep = float(element_filter_buffer) * float(r_out)

    skipped_crack = 0
    used_gp = 0

    for e in range(conn.shape[0]):
        nodes = conn[e]
        xe = pts[nodes, :]                   # (4,2)
        xc = xe.mean(axis=0)
        if np.linalg.norm(xc - tip) > r_keep:
            continue

        ue = np.column_stack([U[2*nodes], U[2*nodes + 1]])  # (4,2)

        for (xi, eta) in gps:
            xgp, detJ, dN_dx, grad_u = q4_kinematics_at_gp(xe, ue, xi, eta)

            # ---- new: exclude Gauss points too close to crack faces ----
            if exclude_crack_faces:
                dseg = point_segment_distance(xgp, crack_start, crack_end)
                if dseg < crack_face_exclusion:
                    skipped_crack += 1
                    continue

            # ---- patched: smoothstep q-field ----
            q, gradq = q_field_radial_smoothstep(xgp, tip, r_in, r_out)
            if gradq[0] == 0.0 and gradq[1] == 0.0:
                continue  # outside annulus or inside r_in where gradq=0

            # strain in Voigt: [exx, eyy, gxy] with gxy = du/dy + dv/dx
            exx = grad_u[0, 0]
            eyy = grad_u[1, 1]
            gxy = grad_u[0, 1] + grad_u[1, 0]
            eps = np.array([exx, eyy, gxy], dtype=float)

            # stress Voigt: [sxx, syy, txy]
            sig = D @ eps
            sxx, syy, txy = float(sig[0]), float(sig[1]), float(sig[2])

            # strain energy density
            Wdens = 0.5 * float(sig @ eps)

            # u_{i,1} along crack_dir:
            du_ds = float(grad_u[0, :] @ crack_dir)
            dv_ds = float(grad_u[1, :] @ crack_dir)

            # term_j = (σ_ij u_{i,1} - W δ_{1j})
            term = np.array([
                sxx * du_ds + txy * dv_ds - Wdens,
                txy * du_ds + syy * dv_ds
            ], dtype=float)

            integrand = float(term @ gradq)
            Jval += integrand * detJ * w * w
            used_gp += 1

    Ep = E_prime(E, nu, plane_stress)
    KI = float(np.sqrt(max(Jval, 0.0) * Ep))

    if log:
        logging.info(
            f"[J] tip={tip}, r_in={r_in}, r_out={r_out}, "
            f"J={Jval:.6e} N/m, KI={KI:.6e} Pa*sqrt(m), "
            f"used_gp={used_gp}, skipped_crack_gp={skipped_crack}, "
            f"crack_face_exclusion={crack_face_exclusion:.3e} m"
        )

    return float(Jval), float(KI)


# ----------------------------
# Convenience: sweep outer radius for path-independence check
# ----------------------------
@dataclass
class JSweepResult:
    r_in: float
    r_out: float
    J: float
    KI: float


def sweep_J_rout(
    pts: np.ndarray,
    conn: np.ndarray,
    U: np.ndarray,
    tip: np.ndarray,
    E: float,
    nu: float,
    plane_stress: bool,
    r_in: float,
    r_out_list: List[float],
    crack_dir: np.ndarray = np.array([1.0, 0.0], dtype=float),
    log_each: bool = True,
    # pass-through exclusions
    crack_start: Optional[np.ndarray] = None,
    crack_end: Optional[np.ndarray] = None,
    exclude_crack_faces: bool = True,
    crack_face_exclusion: Optional[float] = None,
) -> List[JSweepResult]:
    out: List[JSweepResult] = []
    for r_out in r_out_list:
        if r_out <= r_in:
            raise ValueError("Each r_out must be > r_in.")
        J, KI = compute_J_domain_q4(
            pts=pts, conn=conn, U=U, tip=tip,
            E=E, nu=nu, plane_stress=plane_stress,
            r_in=r_in, r_out=r_out,
            crack_dir=crack_dir,
            log=log_each,
            crack_start=crack_start,
            crack_end=crack_end,
            exclude_crack_faces=exclude_crack_faces,
            crack_face_exclusion=crack_face_exclusion,
        )
        out.append(JSweepResult(r_in=r_in, r_out=r_out, J=J, KI=KI))
    return out


# ----------------------------
# Example usage (call from FEM solve script)
# ----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Import this module and call compute_J_domain_q4(...) from your FEM solve script.")
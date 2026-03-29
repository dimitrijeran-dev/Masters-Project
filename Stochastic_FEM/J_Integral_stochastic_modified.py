#!/usr/bin/env python3
"""
J_Integral_stochastic_modified.py

Fracture post-processing utilities for stochastic / heterogeneous Q4 meshes.

Implemented methods
-------------------
1. Corrected J* integral (Eischen-style inhomogeneity correction for varying E)
2. Corrected interaction integral for Mode I / Mode II extraction

Recommended use
---------------
For the current repository, the corrected J* path is the primary production
extractor for stochastic runs. The interaction-integral utilities are retained
for benchmarking and future extension, but should still be validated on a clean
homogeneous Mode-I benchmark before being used as the primary reportable result.

Key implementation change in this rewrite
-----------------------------------------
Elementwise stochastic E fields are first projected to a continuous nodal field.
The Gauss-point modulus Egp and gradient ∇E are then evaluated from that
projected field. This prevents the heterogeneous correction from collapsing to
zero when E_elem is piecewise constant per element.

Scope / assumptions
-------------------
- 2D isotropic linear elasticity
- Q4 bilinear elements
- Straight crack aligned with the provided crack_dir
- Spatial variability only through Young's modulus E(x)
- Poisson's ratio is taken constant
- Auxiliary fields are classical unit-K Williams fields evaluated in the
  crack-aligned local frame

Notes
-----
- The auxiliary displacement gradients are evaluated by stable finite
  differencing of the closed-form asymptotic displacement field.
- Engineering-strain Voigt notation is used consistently:
    [eps_xx, eps_yy, gamma_xy], [sig_xx, sig_yy, tau_xy]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Constitutive helpers
# -----------------------------------------------------------------------------

def D_matrix(E: float, nu: float, plane_stress: bool) -> np.ndarray:
    if plane_stress:
        c = E / (1.0 - nu ** 2)
        return c * np.array(
            [[1.0, nu, 0.0],
             [nu, 1.0, 0.0],
             [0.0, 0.0, (1.0 - nu) / 2.0]],
            dtype=float,
        )

    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array(
        [[1.0 - nu, nu, 0.0],
         [nu, 1.0 - nu, 0.0],
         [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]],
        dtype=float,
    )


def dS_dE_matrix(E: float, nu: float, plane_stress: bool) -> np.ndarray:
    if plane_stress:
        return np.array(
            [[-1.0 / (E * E),  nu / (E * E), 0.0],
             [ nu / (E * E), -1.0 / (E * E), 0.0],
             [0.0, 0.0, -2.0 * (1.0 + nu) / (E * E)]],
            dtype=float,
        )
    return np.array(
        [[-1.0 / (E * E),  nu / (E * E), 0.0],
         [ nu / (E * E), -1.0 / (E * E), 0.0],
         [0.0, 0.0, -2.0 * (1.0 + nu) / (E * E)]],
        dtype=float,
    )


def E_prime(E: float, nu: float, plane_stress: bool) -> float:
    return float(E if plane_stress else E / (1.0 - nu ** 2))


def shear_modulus(E: float, nu: float) -> float:
    return float(E / (2.0 * (1.0 + nu)))


def kappa(E: float, nu: float, plane_stress: bool) -> float:
    return float((3.0 - nu) / (1.0 + nu) if plane_stress else 3.0 - 4.0 * nu)


# -----------------------------------------------------------------------------
# Mesh / kinematics helpers
# -----------------------------------------------------------------------------

def resolve_elementwise_E(
    ne: int,
    E: Optional[float] = None,
    E_elem: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    if E_elem is not None:
        E_arr = np.asarray(E_elem, dtype=float).reshape(-1)
        if E_arr.size != ne:
            raise ValueError(f"E_elem length {E_arr.size} does not match number of elements {ne}.")
        return E_arr, float(np.mean(E_arr))

    if E is None:
        raise ValueError("Provide either scalar E or elementwise E_elem.")

    return np.full(ne, float(E), dtype=float), float(E)


def q4_shape(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
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
            [ 0.25 * (1 - eta), -0.25 * (1 + xi)],
            [ 0.25 * (1 + eta),  0.25 * (1 + xi)],
            [-0.25 * (1 + eta),  0.25 * (1 - xi)],
        ],
        dtype=float,
    )
    return N, dN_dxi


def q4_kinematics_at_gp(
    xe: np.ndarray,
    ue: np.ndarray,
    xi: float,
    eta: float,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    xgp : (2,)
    detJ : float
    N : (4,)
    dN_dx : (4,2)
    grad_u : (2,2)
    """
    N, dN_dxi = q4_shape(xi, eta)
    J = xe.T @ dN_dxi
    detJ = float(np.linalg.det(J))
    if detJ <= 0.0:
        raise ValueError(f"Non-positive detJ={detJ}. Element may be inverted.")
    invJ = np.linalg.inv(J)
    dN_dx = dN_dxi @ invJ.T
    xgp = N @ xe
    grad_u = ue.T @ dN_dx
    return xgp, detJ, N, dN_dx, grad_u


def element_area_q4(xe: np.ndarray) -> float:
    g = 1.0 / np.sqrt(3.0)
    gps = [(-g, -g), (g, -g), (g, g), (-g, g)]
    area = 0.0
    for xi, eta in gps:
        _, dN_dxi = q4_shape(xi, eta)
        J = xe.T @ dN_dxi
        area += abs(float(np.linalg.det(J)))
    return float(area)


def build_projected_nodal_E(pts: np.ndarray, conn: np.ndarray, E_by_elem: np.ndarray) -> np.ndarray:
    nnode = pts.shape[0]
    E_acc = np.zeros(nnode, dtype=float)
    w_acc = np.zeros(nnode, dtype=float)

    for e, nodes in enumerate(conn):
        xe = pts[nodes, :]
        area = max(element_area_q4(xe), 1e-30)
        for nid in nodes:
            E_acc[int(nid)] += area * float(E_by_elem[e])
            w_acc[int(nid)] += area

    E_nodal = np.copy(E_acc)
    valid = w_acc > 0.0
    E_nodal[valid] /= w_acc[valid]
    if not np.all(valid):
        # fall back to nearest elementwise average where needed
        E_nodal[~valid] = float(np.mean(E_by_elem))
    return E_nodal


def E_field_at_gp(N: np.ndarray, dN_dx: np.ndarray, E_nodes: np.ndarray) -> Tuple[float, np.ndarray]:
    E_nodes = np.asarray(E_nodes, dtype=float).reshape(-1)
    Egp = float(N @ E_nodes)
    dE_dx = dN_dx.T @ E_nodes
    return Egp, np.asarray(dE_dx, dtype=float).reshape(2)


# -----------------------------------------------------------------------------
# Geometry / frame helpers
# -----------------------------------------------------------------------------

def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
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


def build_local_frame(crack_dir: np.ndarray) -> np.ndarray:
    t = np.asarray(crack_dir, dtype=float).reshape(2)
    nrm = np.linalg.norm(t)
    if nrm <= 0.0:
        raise ValueError("crack_dir must be nonzero.")
    t = t / nrm
    n = np.array([-t[1], t[0]], dtype=float)
    return np.column_stack([t, n])  # columns: tangent, normal


def to_local(x: np.ndarray, tip: np.ndarray, R: np.ndarray) -> np.ndarray:
    return R.T @ (np.asarray(x, dtype=float).reshape(2) - np.asarray(tip, dtype=float).reshape(2))


def q_field_radial_smoothstep(x: np.ndarray, tip: np.ndarray, r_in: float, r_out: float) -> Tuple[float, np.ndarray]:
    dx = x - tip
    r = float(np.linalg.norm(dx))

    if r <= r_in:
        return 1.0, np.zeros(2, dtype=float)
    if r >= r_out:
        return 0.0, np.zeros(2, dtype=float)

    dr = r_out - r_in
    if dr <= 0.0:
        raise ValueError("r_out must be > r_in")

    xi = (r - r_in) / dr
    q = 1.0 - 3.0 * xi * xi + 2.0 * xi * xi * xi
    dq_dxi = -6.0 * xi + 6.0 * xi * xi
    dq_dr = dq_dxi / dr

    if r < 1e-14:
        return float(q), np.zeros(2, dtype=float)

    gradq = dq_dr * (dx / r)
    return float(q), gradq


# -----------------------------------------------------------------------------
# Auxiliary asymptotic fields
# -----------------------------------------------------------------------------

def _safe_polar(local_x: np.ndarray) -> Tuple[float, float]:
    x, y = float(local_x[0]), float(local_x[1])
    r = max(np.hypot(x, y), 1e-14)
    th = float(np.arctan2(y, x))
    return r, th


def _mode_I_stress_local(local_x: np.ndarray, K: float = 1.0) -> np.ndarray:
    # returns [sxx, syy, txy] in the crack-aligned local CARTESIAN frame
    r, th = _safe_polar(local_x)
    fac = K / np.sqrt(2.0 * np.pi * r)
    c = np.cos(th / 2.0)
    s = np.sin(th / 2.0)
    c3 = np.cos(3.0 * th / 2.0)

    sxx = fac * c * (1.0 - s * np.sin(3.0 * th / 2.0))
    syy = fac * c * (1.0 + s * np.sin(3.0 * th / 2.0))
    txy = fac * s * c * c3
    return np.array([sxx, syy, txy], dtype=float)


def _mode_II_stress_local(local_x: np.ndarray, K: float = 1.0) -> np.ndarray:
    # returns [sxx, syy, txy] in the crack-aligned local CARTESIAN frame
    r, th = _safe_polar(local_x)
    fac = K / np.sqrt(2.0 * np.pi * r)
    s = np.sin(th / 2.0)
    c = np.cos(th / 2.0)
    c3 = np.cos(3.0 * th / 2.0)

    sxx = -fac * s * (2.0 + c * c3)
    syy =  fac * s * c * c3
    txy =  fac * c * (1.0 - s * np.sin(3.0 * th / 2.0))
    return np.array([sxx, syy, txy], dtype=float)


def _mode_I_displacement_local(local_x: np.ndarray, E_tip: float, nu_tip: float, plane_stress: bool, K: float = 1.0) -> np.ndarray:
    r, th = _safe_polar(local_x)
    mu = shear_modulus(E_tip, nu_tip)
    kap = kappa(E_tip, nu_tip, plane_stress)
    fac = K / (2.0 * mu) * np.sqrt(r / (2.0 * np.pi))
    ux = fac * np.cos(th / 2.0) * (kap - np.cos(th))
    uy = fac * np.sin(th / 2.0) * (kap - np.cos(th))
    return np.array([ux, uy], dtype=float)


def _mode_II_displacement_local(local_x: np.ndarray, E_tip: float, nu_tip: float, plane_stress: bool, K: float = 1.0) -> np.ndarray:
    r, th = _safe_polar(local_x)
    mu = shear_modulus(E_tip, nu_tip)
    kap = kappa(E_tip, nu_tip, plane_stress)
    fac = K / (2.0 * mu) * np.sqrt(r / (2.0 * np.pi))
    ux = fac * np.sin(th / 2.0) * (kap + 2.0 + np.cos(th))
    uy = -fac * np.cos(th / 2.0) * (kap - 2.0 + np.cos(th))
    return np.array([ux, uy], dtype=float)


def _rotate_stress_local_to_global(sig_local: np.ndarray, R: np.ndarray) -> np.ndarray:
    S_local = np.array(
        [[sig_local[0], sig_local[2]],
         [sig_local[2], sig_local[1]]],
        dtype=float,
    )
    S_global = R @ S_local @ R.T
    return np.array([S_global[0, 0], S_global[1, 1], S_global[0, 1]], dtype=float)


def _aux_displacement_global(
    x_global: np.ndarray,
    tip: np.ndarray,
    R: np.ndarray,
    mode: str,
    E_tip: float,
    nu_tip: float,
    plane_stress: bool,
) -> np.ndarray:
    x_local = to_local(x_global, tip, R)
    if mode == "I":
        u_local = _mode_I_displacement_local(x_local, E_tip, nu_tip, plane_stress, K=1.0)
    elif mode == "II":
        u_local = _mode_II_displacement_local(x_local, E_tip, nu_tip, plane_stress, K=1.0)
    else:
        raise ValueError("mode must be 'I' or 'II'")
    return R @ u_local


def _aux_stress_global(
    x_global: np.ndarray,
    tip: np.ndarray,
    R: np.ndarray,
    mode: str,
    E_tip: float,
    nu_tip: float,
    plane_stress: bool,
) -> np.ndarray:
    del E_tip, nu_tip, plane_stress
    x_local = to_local(x_global, tip, R)
    if mode == "I":
        sig_local = _mode_I_stress_local(x_local, K=1.0)
    elif mode == "II":
        sig_local = _mode_II_stress_local(x_local, K=1.0)
    else:
        raise ValueError("mode must be 'I' or 'II'")
    return _rotate_stress_local_to_global(sig_local, R)


def _aux_grad_u_global(
    x_global: np.ndarray,
    tip: np.ndarray,
    R: np.ndarray,
    mode: str,
    E_tip: float,
    nu_tip: float,
    plane_stress: bool,
) -> np.ndarray:
    local_x = to_local(x_global, tip, R)
    r = max(float(np.linalg.norm(local_x)), 1e-12)
    h = max(1e-8, min(1e-5, 1e-2 * r))

    ex = np.array([h, 0.0], dtype=float)
    ey = np.array([0.0, h], dtype=float)

    ux_p = _aux_displacement_global(x_global + ex, tip, R, mode, E_tip, nu_tip, plane_stress)
    ux_m = _aux_displacement_global(x_global - ex, tip, R, mode, E_tip, nu_tip, plane_stress)
    uy_p = _aux_displacement_global(x_global + ey, tip, R, mode, E_tip, nu_tip, plane_stress)
    uy_m = _aux_displacement_global(x_global - ey, tip, R, mode, E_tip, nu_tip, plane_stress)

    dudx = (ux_p - ux_m) / (2.0 * h)
    dudy = (uy_p - uy_m) / (2.0 * h)
    return np.column_stack([dudx, dudy])


# -----------------------------------------------------------------------------
# Corrected J* integral
# -----------------------------------------------------------------------------

def compute_J_domain_q4(
    pts: np.ndarray,
    conn: np.ndarray,
    U: np.ndarray,
    tip: np.ndarray,
    E: Optional[float] = None,
    nu: float = 0.33,
    plane_stress: bool = True,
    r_in: float = 0.008,
    r_out: float = 0.020,
    crack_dir: np.ndarray = np.array([1.0, 0.0], dtype=float),
    element_filter_buffer: float = 1.5,
    log: bool = False,
    crack_start: Optional[np.ndarray] = None,
    crack_end: Optional[np.ndarray] = None,
    exclude_crack_faces: bool = True,
    crack_face_exclusion: Optional[float] = None,
    E_elem: Optional[np.ndarray] = None,
    E_for_KI: Optional[float] = None,
    use_eischen_correction: bool = True,
) -> Tuple[float, float]:
    pts = np.asarray(pts, dtype=float)
    conn = np.asarray(conn, dtype=int)
    U = np.asarray(U, dtype=float).reshape(-1)
    tip = np.asarray(tip, dtype=float).reshape(2)

    if crack_start is None:
        crack_start = np.array([0.0, 0.0], dtype=float)
    else:
        crack_start = np.asarray(crack_start, dtype=float).reshape(2)

    if crack_end is None:
        crack_end = tip.copy()
    else:
        crack_end = np.asarray(crack_end, dtype=float).reshape(2)

    if crack_face_exclusion is None:
        crack_face_exclusion = 3e-4 * float(r_out)
    crack_face_exclusion = float(crack_face_exclusion)

    crack_dir = np.asarray(crack_dir, dtype=float).reshape(2)
    crack_dir = crack_dir / np.linalg.norm(crack_dir)

    ne = conn.shape[0]
    E_by_elem, E_scalar_default = resolve_elementwise_E(ne=ne, E=E, E_elem=E_elem)
    if E_for_KI is None:
        E_for_KI = E_scalar_default
    E_for_KI = float(E_for_KI)

    E_nodal = build_projected_nodal_E(pts, conn, E_by_elem) if (E_elem is not None and use_eischen_correction) else None

    J_standard = 0.0
    J_correction = 0.0
    r_keep = float(element_filter_buffer) * float(r_out)
    skipped_crack = 0
    used_gp = 0

    g = 1.0 / np.sqrt(3.0)
    gps = [(-g, -g), (g, -g), (g, g), (-g, g)]

    for e in range(ne):
        nodes = conn[e]
        xe = pts[nodes, :]
        xc = xe.mean(axis=0)
        if np.linalg.norm(xc - tip) > r_keep:
            continue

        ue = np.column_stack([U[2 * nodes], U[2 * nodes + 1]])

        for (xi, eta) in gps:
            xgp, detJ, N, dN_dx, grad_u = q4_kinematics_at_gp(xe, ue, xi, eta)

            if exclude_crack_faces:
                dseg = point_segment_distance(xgp, crack_start, crack_end)
                if dseg < crack_face_exclusion:
                    skipped_crack += 1
                    continue

            q, gradq = q_field_radial_smoothstep(xgp, tip, r_in, r_out)
            if gradq[0] == 0.0 and gradq[1] == 0.0 and q == 0.0:
                continue

            if E_nodal is not None:
                Egp, dE_dx_vec = E_field_at_gp(N, dN_dx, E_nodal[nodes])
                Egp = max(Egp, 1e-30)
            else:
                Egp = float(E_by_elem[e])
                dE_dx_vec = np.zeros(2, dtype=float)

            D = D_matrix(Egp, nu, plane_stress)

            exx = grad_u[0, 0]
            eyy = grad_u[1, 1]
            gxy = grad_u[0, 1] + grad_u[1, 0]
            eps = np.array([exx, eyy, gxy], dtype=float)

            sig = D @ eps
            sxx, syy, txy = map(float, sig)
            Wdens = 0.5 * float(sig @ eps)

            du_ds = float(grad_u[0, :] @ crack_dir)
            dv_ds = float(grad_u[1, :] @ crack_dir)
            term = np.array(
                [sxx * du_ds + txy * dv_ds - Wdens,
                 txy * du_ds + syy * dv_ds],
                dtype=float,
            )
            J_standard += float(term @ gradq) * detJ

            if E_nodal is not None:
                dE_ds = float(dE_dx_vec @ crack_dir)
                J_correction += (-(Wdens / Egp) * dE_ds * q) * detJ

            used_gp += 1

    J_total = float(J_standard + J_correction)
    KI = float(np.sqrt(max(J_total, 0.0) * E_prime(E_for_KI, nu, plane_stress)))

    if log:
        logging.info(
            "[J*] tip=%s, r_in=%.6g, r_out=%.6g, Jstd=%.6e, Jcorr=%.6e, J*=%.6e, KI=%.6e, used_gp=%d, skipped_crack_gp=%d",
            tip, r_in, r_out, J_standard, J_correction, J_total, KI, used_gp, skipped_crack,
        )

    return J_total, KI


# -----------------------------------------------------------------------------
# Corrected interaction integral
# -----------------------------------------------------------------------------

@dataclass
class InteractionResult:
    r_in: float
    r_out: float
    M_I: float
    KI: float
    M_II: float
    KII: float


@dataclass
class JSweepResult:
    r_in: float
    r_out: float
    J: float
    KI: float


@dataclass
class _ModeAccum:
    M_standard: float = 0.0
    M_correction: float = 0.0


def compute_interaction_integral_q4(
    pts: np.ndarray,
    conn: np.ndarray,
    U: np.ndarray,
    tip: np.ndarray,
    E: Optional[float] = None,
    nu: float = 0.33,
    plane_stress: bool = True,
    r_in: float = 0.008,
    r_out: float = 0.020,
    crack_dir: np.ndarray = np.array([1.0, 0.0], dtype=float),
    element_filter_buffer: float = 1.5,
    log: bool = False,
    crack_start: Optional[np.ndarray] = None,
    crack_end: Optional[np.ndarray] = None,
    exclude_crack_faces: bool = True,
    crack_face_exclusion: Optional[float] = None,
    E_elem: Optional[np.ndarray] = None,
    E_tip: Optional[float] = None,
    modes: Tuple[str, ...] = ("I", "II"),
    use_inhomogeneity_correction: bool = True,
) -> dict:
    pts = np.asarray(pts, dtype=float)
    conn = np.asarray(conn, dtype=int)
    U = np.asarray(U, dtype=float).reshape(-1)
    tip = np.asarray(tip, dtype=float).reshape(2)

    if crack_start is None:
        crack_start = np.array([0.0, 0.0], dtype=float)
    else:
        crack_start = np.asarray(crack_start, dtype=float).reshape(2)

    if crack_end is None:
        crack_end = tip.copy()
    else:
        crack_end = np.asarray(crack_end, dtype=float).reshape(2)

    if crack_face_exclusion is None:
        crack_face_exclusion = 3e-4 * float(r_out)
    crack_face_exclusion = float(crack_face_exclusion)

    crack_dir = np.asarray(crack_dir, dtype=float).reshape(2)
    crack_dir = crack_dir / np.linalg.norm(crack_dir)
    R = build_local_frame(crack_dir)

    ne = conn.shape[0]
    E_by_elem, E_scalar_default = resolve_elementwise_E(ne=ne, E=E, E_elem=E_elem)
    if E_tip is None:
        E_tip = E_scalar_default
    E_tip = float(E_tip)
    Ep_tip = E_prime(E_tip, nu, plane_stress)

    E_nodal = build_projected_nodal_E(pts, conn, E_by_elem) if E_elem is not None else None

    acc = {mode: _ModeAccum() for mode in modes}
    r_keep = float(element_filter_buffer) * float(r_out)
    skipped_crack = 0
    used_gp = 0

    g = 1.0 / np.sqrt(3.0)
    gps = [(-g, -g), (g, -g), (g, g), (-g, g)]

    for e in range(ne):
        nodes = conn[e]
        xe = pts[nodes, :]
        xc = xe.mean(axis=0)
        if np.linalg.norm(xc - tip) > r_keep:
            continue

        ue = np.column_stack([U[2 * nodes], U[2 * nodes + 1]])

        for (xi, eta) in gps:
            xgp, detJ, N, dN_dx, grad_u = q4_kinematics_at_gp(xe, ue, xi, eta)

            if exclude_crack_faces:
                dseg = point_segment_distance(xgp, crack_start, crack_end)
                if dseg < crack_face_exclusion:
                    skipped_crack += 1
                    continue

            q, gradq = q_field_radial_smoothstep(xgp, tip, r_in, r_out)
            if gradq[0] == 0.0 and gradq[1] == 0.0 and q == 0.0:
                continue

            if E_nodal is not None:
                Egp, dE_dx_vec = E_field_at_gp(N, dN_dx, E_nodal[nodes])
                Egp = max(Egp, 1e-30)
            else:
                Egp = float(E_by_elem[e])
                dE_dx_vec = np.zeros(2, dtype=float)

            D = D_matrix(Egp, nu, plane_stress)

            eps = np.array(
                [grad_u[0, 0], grad_u[1, 1], grad_u[0, 1] + grad_u[1, 0]],
                dtype=float,
            )
            sig = D @ eps
            sxx, syy, txy = map(float, sig)

            du_ds = float(grad_u[0, :] @ crack_dir)
            dv_ds = float(grad_u[1, :] @ crack_dir)

            dE_ds = float(dE_dx_vec @ crack_dir)
            dS_ds = dS_dE_matrix(Egp, nu, plane_stress) * dE_ds

            for mode in modes:
                sig_aux = _aux_stress_global(xgp, tip, R, mode, E_tip, nu, plane_stress)
                grad_u_aux = _aux_grad_u_global(xgp, tip, R, mode, E_tip, nu, plane_stress)

                duaux_ds = float(grad_u_aux[0, :] @ crack_dir)
                dvaux_ds = float(grad_u_aux[1, :] @ crack_dir)

                W_int = float(sig_aux[0] * eps[0] + sig_aux[1] * eps[1] + sig_aux[2] * eps[2])
                term = np.array(
                    [
                        sxx * duaux_ds + txy * dvaux_ds + sig_aux[0] * du_ds + sig_aux[2] * dv_ds - W_int,
                        txy * duaux_ds + syy * dvaux_ds + sig_aux[2] * du_ds + sig_aux[1] * dv_ds,
                    ],
                    dtype=float,
                )

                acc[mode].M_standard += float(term @ gradq) * detJ

                if use_inhomogeneity_correction and E_nodal is not None:
                    acc[mode].M_correction += float(sig_aux @ (dS_ds @ sig)) * q * detJ

            used_gp += 1

    out: dict = {}
    if "I" in modes:
        M_I = float(acc["I"].M_standard + acc["I"].M_correction)
        KI = float(0.5 * Ep_tip * M_I)
        out["M_I_standard"] = float(acc["I"].M_standard)
        out["M_I_correction"] = float(acc["I"].M_correction)
        out["M_I"] = M_I
        out["KI"] = KI
    if "II" in modes:
        M_II = float(acc["II"].M_standard + acc["II"].M_correction)
        KII = float(0.5 * Ep_tip * M_II)
        out["M_II_standard"] = float(acc["II"].M_standard)
        out["M_II_correction"] = float(acc["II"].M_correction)
        out["M_II"] = M_II
        out["KII"] = KII

    if log:
        logging.info(
            "[Interaction] tip=%s, r_in=%.6g, r_out=%.6g, results=%s, used_gp=%d, skipped_crack_gp=%d",
            tip, r_in, r_out, {k: float(v) for k, v in out.items()}, used_gp, skipped_crack
        )

    return out


# -----------------------------------------------------------------------------
# Sweep helpers
# -----------------------------------------------------------------------------

def sweep_J_rout(
    pts: np.ndarray,
    conn: np.ndarray,
    U: np.ndarray,
    tip: np.ndarray,
    E: Optional[float] = None,
    nu: float = 0.33,
    plane_stress: bool = True,
    r_in: float = 0.008,
    r_out_list: Optional[List[float]] = None,
    crack_dir: np.ndarray = np.array([1.0, 0.0], dtype=float),
    log_each: bool = True,
    crack_start: Optional[np.ndarray] = None,
    crack_end: Optional[np.ndarray] = None,
    exclude_crack_faces: bool = True,
    crack_face_exclusion: Optional[float] = None,
    E_elem: Optional[np.ndarray] = None,
    E_for_KI: Optional[float] = None,
    use_eischen_correction: bool = True,
) -> List[JSweepResult]:
    if r_out_list is None:
        r_out_list = [0.016, 0.018, 0.020]

    out: List[JSweepResult] = []
    for r_out in r_out_list:
        if r_out <= r_in:
            raise ValueError("Each r_out must be > r_in.")
        J, KI = compute_J_domain_q4(
            pts=pts,
            conn=conn,
            U=U,
            tip=tip,
            E=E,
            E_elem=E_elem,
            E_for_KI=E_for_KI,
            nu=nu,
            plane_stress=plane_stress,
            r_in=r_in,
            r_out=r_out,
            crack_dir=crack_dir,
            log=log_each,
            crack_start=crack_start,
            crack_end=crack_end,
            exclude_crack_faces=exclude_crack_faces,
            crack_face_exclusion=crack_face_exclusion,
            use_eischen_correction=use_eischen_correction,
        )
        out.append(JSweepResult(r_in=r_in, r_out=r_out, J=J, KI=KI))
    return out


def sweep_interaction_rout(
    pts: np.ndarray,
    conn: np.ndarray,
    U: np.ndarray,
    tip: np.ndarray,
    E: Optional[float] = None,
    nu: float = 0.33,
    plane_stress: bool = True,
    r_in: float = 0.008,
    r_out_list: Optional[List[float]] = None,
    crack_dir: np.ndarray = np.array([1.0, 0.0], dtype=float),
    log_each: bool = True,
    crack_start: Optional[np.ndarray] = None,
    crack_end: Optional[np.ndarray] = None,
    exclude_crack_faces: bool = True,
    crack_face_exclusion: Optional[float] = None,
    E_elem: Optional[np.ndarray] = None,
    E_tip: Optional[float] = None,
    modes: Tuple[str, ...] = ("I", "II"),
    use_inhomogeneity_correction: bool = True,
) -> List[InteractionResult]:
    if r_out_list is None:
        r_out_list = [0.016, 0.018, 0.020]

    out: List[InteractionResult] = []
    for r_out in r_out_list:
        if r_out <= r_in:
            raise ValueError("Each r_out must be > r_in.")
        res = compute_interaction_integral_q4(
            pts=pts,
            conn=conn,
            U=U,
            tip=tip,
            E=E,
            nu=nu,
            plane_stress=plane_stress,
            r_in=r_in,
            r_out=r_out,
            crack_dir=crack_dir,
            log=log_each,
            crack_start=crack_start,
            crack_end=crack_end,
            exclude_crack_faces=exclude_crack_faces,
            crack_face_exclusion=crack_face_exclusion,
            E_elem=E_elem,
            E_tip=E_tip,
            modes=modes,
            use_inhomogeneity_correction=use_inhomogeneity_correction,
        )
        out.append(
            InteractionResult(
                r_in=r_in,
                r_out=r_out,
                M_I=float(res.get("M_I", np.nan)),
                KI=float(res.get("KI", np.nan)),
                M_II=float(res.get("M_II", np.nan)),
                KII=float(res.get("KII", np.nan)),
            )
        )
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Import this module and call compute_J_domain_q4(...) or compute_interaction_integral_q4(...)")

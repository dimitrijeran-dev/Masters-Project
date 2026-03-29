import numpy as np

from src.J_Integral import (
    D_matrix,
    E_prime,
    point_segment_distance,
    q4_kinematics_at_gp,
    q4_shape,
    q_field_radial_smoothstep,
)


def test_constitutive_helpers_basic_sanity():
    E = 210e9
    nu = 0.3

    d_ps = D_matrix(E, nu, plane_stress=True)
    d_pe = D_matrix(E, nu, plane_stress=False)

    assert d_ps.shape == (3, 3)
    assert d_pe.shape == (3, 3)
    assert np.all(np.isfinite(d_ps))
    assert np.all(np.isfinite(d_pe))
    assert E_prime(E, nu, plane_stress=True) == E
    assert np.isclose(E_prime(E, nu, plane_stress=False), E / (1 - nu**2))


def test_q4_shape_and_kinematics_on_unit_square():
    N, dN_dxi = q4_shape(0.0, 0.0)
    assert np.isclose(N.sum(), 1.0)
    assert dN_dxi.shape == (4, 2)

    xe = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    ue = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]])  # u=x, v=2y

    xgp, detJ, dN_dx, grad_u = q4_kinematics_at_gp(xe, ue, 0.0, 0.0)

    assert np.allclose(xgp, [0.5, 0.5])
    assert detJ > 0
    assert dN_dx.shape == (4, 2)
    assert np.allclose(grad_u, [[1.0, 0.0], [0.0, 2.0]], atol=1e-12)


def test_distance_and_smoothstep_q_field_sanity():
    p = np.array([0.5, 1.0])
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    assert np.isclose(point_segment_distance(p, a, b), 1.0)

    tip = np.array([0.0, 0.0])
    q_in, g_in = q_field_radial_smoothstep(np.array([0.1, 0.0]), tip, 0.2, 0.5)
    q_mid, g_mid = q_field_radial_smoothstep(np.array([0.3, 0.0]), tip, 0.2, 0.5)
    q_out, g_out = q_field_radial_smoothstep(np.array([0.8, 0.0]), tip, 0.2, 0.5)

    assert q_in == 1.0 and np.allclose(g_in, 0.0)
    assert 0.0 < q_mid < 1.0
    assert np.all(np.isfinite(g_mid))
    assert q_out == 0.0 and np.allclose(g_out, 0.0)

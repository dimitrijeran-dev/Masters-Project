import numpy as np

from Lifing.fatigue_lifing_utils import (
    compute_delta_k,
    compute_delta_k_from_R,
    integrate_crack_growth,
    paris_law,
)


def test_delta_k_helpers_agree():
    kmax = np.array([10.0, 12.0, 15.0])
    r = 0.2
    kmin = r * kmax

    direct = compute_delta_k(kmax, kmin)
    via_r = compute_delta_k_from_R(kmax, r)

    assert np.allclose(direct, via_r)
    assert np.all(direct > 0.0)


def test_paris_law_monotonicity_for_positive_m():
    delta_k = np.array([5.0, 10.0, 20.0])
    dadn = paris_law(delta_k, C=1e-12, m=3.0)

    assert np.all(np.diff(dadn) > 0.0)


def test_integrate_crack_growth_shape_and_finite_values():
    a = np.linspace(0.01, 0.02, 11)
    delta_k = np.linspace(8.0, 12.0, 11)

    N, dadn = integrate_crack_growth(a, delta_k, C=1e-11, m=2.5)

    assert N.shape == a.shape
    assert dadn.shape == a.shape
    assert np.all(np.isfinite(N))
    assert np.all(np.isfinite(dadn))
    assert np.all(np.diff(N) > 0.0)

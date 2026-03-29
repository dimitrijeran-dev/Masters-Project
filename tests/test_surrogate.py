import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
np = pytest.importorskip("numpy")

from UQ.surrogate import fit_polynomial_chaos, surrogate_mean_variance


def test_surrogate_recovers_polynomial_and_moments():
    xi = np.linspace(-2, 2, 11).reshape(-1, 1)
    y = 2.0 + 3.0 * xi[:, 0] + 0.5 * xi[:, 0] ** 2
    fit = fit_polynomial_chaos(xi, y, order=2)
    pred = np.polynomial.polynomial.polyval(xi[:, 0], [fit.coefficients[0], fit.coefficients[1], fit.coefficients[2]])
    assert np.max(np.abs(pred - y)) < 1e-8
    stats = surrogate_mean_variance(y, np.ones_like(y))
    assert stats["variance"] > 0.0

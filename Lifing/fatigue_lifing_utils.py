import numpy as np
import pandas as pd


def load_ki_csv(path):
    df = pd.read_csv(path)
    if "a" not in df.columns or "KI" not in df.columns:
        raise ValueError("CSV must contain columns: a, KI")
    return df.sort_values("a").reset_index(drop=True)


def compute_delta_k_from_R(Kmax, R):
    return Kmax * (1.0 - R)


def compute_delta_k(Kmax, Kmin):
    return Kmax - Kmin


def paris_law(delta_k, C, m):
    return C * (delta_k ** m)


def integrate_crack_growth(a, delta_k, C, m):
    """
    Integrate dN = da / (da/dN) using the supplied a-grid.
    Returns:
        N : cumulative cycles aligned with a
        dadn : crack growth rate aligned with a
    """
    a = np.asarray(a, dtype=float)
    delta_k = np.asarray(delta_k, dtype=float)

    dadn = paris_law(delta_k, C, m)
    da = np.gradient(a)
    dN = da / dadn
    N = np.cumsum(dN)

    # shift so first point is N=0
    if len(N) > 0:
        N = N - N[0]

    return N, dadn


def find_critical_crack_length(a, kmax, kic):
    """
    Find a_crit from first crossing of Kmax(a) = KIC.
    Uses linear interpolation between bracketing points.

    Returns:
        acrit : float or None
        idx_fail : int or None
            index of first point where Kmax >= KIC
    """
    a = np.asarray(a, dtype=float)
    kmax = np.asarray(kmax, dtype=float)

    if len(a) != len(kmax):
        raise ValueError("a and kmax must have the same length")

    crossed = np.where(kmax >= kic)[0]
    if len(crossed) == 0:
        return None, None

    idx = int(crossed[0])

    # exact first-point failure
    if idx == 0:
        return float(a[0]), idx

    a1, a2 = a[idx - 1], a[idx]
    k1, k2 = kmax[idx - 1], kmax[idx]

    if np.isclose(k2, k1):
        return float(a2), idx

    acrit = a1 + (kic - k1) * (a2 - a1) / (k2 - k1)
    return float(acrit), idx


def truncate_at_acrit(a, y, acrit):
    """
    Truncate a curve y(a) at acrit and append an interpolated endpoint.
    """
    a = np.asarray(a, dtype=float)
    y = np.asarray(y, dtype=float)

    if acrit is None:
        return a, y

    if acrit <= a[0]:
        return np.array([acrit]), np.array([y[0]])

    keep = a < acrit
    a_keep = a[keep]
    y_keep = y[keep]

    ycrit = np.interp(acrit, a, y)

    a_out = np.append(a_keep, acrit)
    y_out = np.append(y_keep, ycrit)
    return a_out, y_out
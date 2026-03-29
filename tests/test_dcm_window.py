import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
np = pytest.importorskip("numpy")

from src.dcm import select_stable_window


def test_select_stable_window_prefers_plateau_middle():
    r = np.linspace(1e-4, 2e-3, 40)
    ki = 30e6 + 2e6 * (r - np.mean(r))
    ki[12:28] = 30e6 + 0.05e6 * np.sin(np.linspace(0, 3.14, 16))
    win = select_stable_window(r, ki, frac_window=0.3, min_points=8)
    assert win["r_min"] > r[5]
    assert win["r_max"] < r[-5]

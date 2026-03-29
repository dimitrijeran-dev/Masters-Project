import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
np = pytest.importorskip("numpy")

from Stochastic_FEM.J_Integral_stochastic_modified import select_stable_jstar_window


def test_select_stable_jstar_window_finds_middle_plateau():
    r_out = np.linspace(0.01, 0.03, 11)
    ki = np.array([31, 30, 29, 28, 27.8, 27.7, 27.8, 27.9, 29, 30, 31], dtype=float) * 1e6
    out = select_stable_jstar_window(r_out, ki, window_size=3)
    assert 3 <= out["selected_index"] <= 7

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
pytest.importorskip("numpy")

from UQ.gpc_types import RandomInput
from UQ.quadrature import tensor_collocation


def test_gauss_hermite_tensor_weights_sum_to_one():
    pts = tensor_collocation([RandomInput(name="E", distribution="normal", mean=1.0, std=0.1, order=3)], order=3)
    assert len(pts) == 3
    wsum = sum(w for _, w in pts)
    assert abs(wsum - 1.0) < 1e-12

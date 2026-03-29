import json
from pathlib import Path


def test_validation_summary_golden_band_check():
    summary_path = Path("Data/New Data/meshrun_20mm/validation_summary.json")
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = {
        "run_name",
        "a",
        "best_r_out",
        "J_ref",
        "KI_ref",
        "J_relative_span",
        "KI_relative_span",
        "JK_relative_residual",
    }
    assert required_keys.issubset(data.keys())

    assert data["run_name"] == "meshrun_0.02mm"
    assert abs(data["a"] - 0.02) < 1e-12

    # Tiny golden tolerance bands to detect accidental drift while being robust to
    # harmless serialization formatting changes.
    assert 3700.0 <= data["J_ref"] <= 3900.0
    assert 1.66e7 <= data["KI_ref"] <= 1.67e7
    assert abs(data["best_r_out"] - 0.0095) < 1e-12
    assert data["J_relative_span"] < 0.03
    assert data["KI_relative_span"] < 0.02
    assert data["JK_relative_residual"] < 1e-12

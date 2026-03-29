import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dcm import CrackFaceDisplacement, Material, estimate_plateau_ki


def _cod_from_ki(ki: float, material: Material, r: float) -> float:
    factor = (2.0 * material.shear_modulus() / (material.kappa() + 1.0)) * math.sqrt(2.0 * math.pi)
    return (ki / factor) * math.sqrt(r)


def test_estimate_plateau_ki_uses_robust_multipoint_fit():
    material = Material(elastic_modulus=73.1e9, poisson_ratio=0.33, plane_strain=False)
    ki_true = 30.0e6

    r_vals = [2.0e-4, 3.0e-4, 4.5e-4, 6.0e-4, 8.0e-4, 1.0e-3, 1.4e-3, 1.8e-3]
    records = []
    for i, r in enumerate(r_vals):
        cod = _cod_from_ki(ki_true, material, r)
        cod *= 1.0 + (0.01 if i % 2 == 0 else -0.01)  # small deterministic perturbation
        records.append(CrackFaceDisplacement(r=r, uy_upper=cod, uy_lower=0.0))

    # Add a severe outlier pair to verify robustness of inlier-filtered fit.
    outlier_cod = 2.8 * _cod_from_ki(ki_true, material, 5.0e-4)
    records.append(CrackFaceDisplacement(r=5.0e-4, uy_upper=outlier_cod, uy_lower=0.0))

    stats = estimate_plateau_ki(records, material, use_median=True)
    ki_fit = float(stats["KI_ref"])

    assert stats["fit_model"] == "cod_vs_sqrt_r_through_origin"
    assert stats["n_samples"] == len(records)
    assert stats["n_inliers"] < stats["n_samples"]
    assert stats["fit_r2"] > 0.95
    assert abs(ki_fit - ki_true) / ki_true < 0.05

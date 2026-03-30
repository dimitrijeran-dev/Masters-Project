#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from UQ.distributions import map_xi_vector_to_parameters
from UQ.gpc_types import CollocationPoint, GPCConfig, QoIResult
from UQ.qoi_extractors import extract_both_qois
from UQ.quadrature import tensor_collocation
from UQ.stats import weighted_moments
from UQ.surrogate import fit_polynomial_chaos
from Stochastic_FEM.stochastic_fem_solver import run_single_realization_with_overrides
from Stochastic_FEM.stochastic_validate_fields import validate_single_realization


def _json_dump(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_gpc_study(config: GPCConfig) -> Dict[str, str]:
    run_dir = config.resolved_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    base_solver_dir = config.solver_run_dir or run_dir
    if not (base_solver_dir / "plate_edge_crack_q4.msh").exists():
        raise FileNotFoundError(f"Missing plate_edge_crack_q4.msh in solver_run_dir={base_solver_dir}")

    points_raw = tensor_collocation(
        config.random_inputs,
        config.quadrature_order,
        allow_uniform=config.include_uniform_tensor_rule,
    )

    collocation_points: List[CollocationPoint] = []
    qoi_rows: List[QoIResult] = []

    for idx, (xi_vec, w) in enumerate(points_raw):
        mapped = map_xi_vector_to_parameters(xi_vec, config.random_inputs)
        cp = CollocationPoint(index=idx, xi=xi_vec, weight=w, mapped_parameters=mapped)
        collocation_points.append(cp)

        case_dir = run_dir / f"case_{idx:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_solver_dir / "plate_edge_crack_q4.msh", case_dir / "mesh_q4.msh")

        rid = config.realization_id_start + idx
        solve_info = run_single_realization_with_overrides(
            run_dir=case_dir,
            realization_id=rid,
            material_overrides=mapped,
            stochastic_field_overrides=None,
            write_outputs=True,
        )
        validation = validate_single_realization(
            run_dir=case_dir,
            realization_id=rid,
            compute_jstar=config.compute_jstar,
            compute_dcm=config.compute_dcm,
            dcm_window=config.dcm_window,
        )
        qois = extract_both_qois(case_dir)
        qoi_rows.append(
            QoIResult(
                case_id=case_dir.name,
                collocation_index=idx,
                xi=list(map(float, xi_vec)),
                weight=float(w),
                KI_Jstar=qois.get("KI_Jstar"),
                KI_DCM=qois.get("KI_DCM"),
                run_dir=str(case_dir),
                artifacts={"solve": solve_info, "validation": validation},
            )
        )

    csv_path = run_dir / "gpc_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        headers = ["case_id", "collocation_index", "weight", "KI_Jstar", "KI_DCM"]
        headers.extend([f"xi_{i}" for i in range(len(config.random_inputs))])
        headers.extend([f"param_{ri.name}" for ri in config.random_inputs])
        w.writerow(headers)
        for cp, row in zip(collocation_points, qoi_rows):
            vals = [row.case_id, row.collocation_index, row.weight, row.KI_Jstar, row.KI_DCM]
            vals.extend(cp.xi)
            vals.extend([cp.mapped_parameters[ri.name] for ri in config.random_inputs])
            w.writerow(vals)

    xi = np.asarray([cp.xi for cp in collocation_points], dtype=float)
    ww = np.asarray([cp.weight for cp in collocation_points], dtype=float)

    summary = {"run_name": config.run_name, "n_collocation": len(collocation_points)}

    for key in ("KI_Jstar", "KI_DCM"):
        y = np.asarray([getattr(r, key) if getattr(r, key) is not None else np.nan for r in qoi_rows], dtype=float)
        mask = np.isfinite(y)
        if np.any(mask):
            fit = fit_polynomial_chaos(xi[mask], y[mask], order=config.polynomial_order)
            coeff_path = run_dir / f"gpc_coefficients_{key}.json"
            _json_dump(
                coeff_path,
                {
                    "qoi": key,
                    "multiindex": [list(a) for a in fit.multiindex],
                    "coefficients": [float(c) for c in fit.coefficients],
                },
            )
            summary[key] = weighted_moments(y[mask], ww[mask])

    _json_dump(run_dir / "gpc_qoi_summary.json", summary)
    _json_dump(run_dir / "gpc_config_resolved.json", asdict(config))
    return {"run_dir": str(run_dir), "samples_csv": str(csv_path), "summary_json": str(run_dir / "gpc_qoi_summary.json")}

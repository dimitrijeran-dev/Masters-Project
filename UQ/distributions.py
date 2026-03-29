from __future__ import annotations

from typing import Dict

from UQ.gpc_types import RandomInput


def map_standard_to_physical(xi: float, rv: RandomInput) -> float:
    dist = rv.distribution.lower()
    if dist == "normal":
        if rv.std is None:
            raise ValueError(f"Random input '{rv.name}' needs std for normal distribution")
        return float(rv.mean + rv.std * xi)
    if dist == "uniform":
        if rv.lower is None or rv.upper is None:
            raise ValueError(f"Random input '{rv.name}' needs lower/upper for uniform distribution")
        # xi expected in [-1,1]
        return float(0.5 * (rv.upper + rv.lower) + 0.5 * (rv.upper - rv.lower) * xi)
    raise ValueError(f"Unsupported distribution: {rv.distribution}")


def map_xi_vector_to_parameters(xi_vec: list[float], random_inputs: list[RandomInput]) -> Dict[str, float]:
    if len(xi_vec) != len(random_inputs):
        raise ValueError("xi vector and random_inputs length mismatch")
    return {
        rv.name: map_standard_to_physical(float(xi), rv)
        for xi, rv in zip(xi_vec, random_inputs)
    }

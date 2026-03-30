#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from UQ.multiindex import total_order_multiindex


@dataclass
class SurrogateFit:
    multiindex: list[tuple[int, ...]]
    coefficients: np.ndarray


def _eval_monomials(xi: np.ndarray, basis: list[tuple[int, ...]]) -> np.ndarray:
    vals = np.ones((xi.shape[0], len(basis)), dtype=float)
    for j, alpha in enumerate(basis):
        for d, p in enumerate(alpha):
            if p:
                vals[:, j] *= xi[:, d] ** p
    return vals


def fit_polynomial_chaos(xi: np.ndarray, y: np.ndarray, order: int) -> SurrogateFit:
    xi = np.asarray(xi, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    basis = total_order_multiindex(xi.shape[1], order)
    A = _eval_monomials(xi, basis)
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    return SurrogateFit(multiindex=basis, coefficients=coeff)


def evaluate_surrogate(fit: SurrogateFit, xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=float)
    A = _eval_monomials(xi, fit.multiindex)
    return A @ fit.coefficients


def surrogate_mean_variance(values: np.ndarray, weights: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    mu = float(np.sum(w * v))
    var = float(np.sum(w * (v - mu) ** 2))
    return {"mean": mu, "variance": var, "std": float(np.sqrt(max(var, 0.0)))}

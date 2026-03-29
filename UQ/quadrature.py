from __future__ import annotations

import itertools
from typing import List, Tuple

import numpy as np

from UQ.gpc_types import RandomInput


def _gh_rule(order: int) -> Tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.hermite.hermgauss(order)
    # Convert for standard normal expectation integral
    xi = np.sqrt(2.0) * x
    ww = w / np.sqrt(np.pi)
    return xi, ww


def _uniform_rule(order: int) -> Tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(order)
    # expectation under U[-1,1]: 0.5 * integral
    return x, 0.5 * w


def tensor_collocation(random_inputs: List[RandomInput], order: int, allow_uniform: bool = True):
    if not random_inputs:
        raise ValueError("At least one random input is required")
    rules = []
    for rv in random_inputs:
        dist = rv.distribution.lower()
        if dist == "normal":
            rules.append(_gh_rule(rv.order if rv.order else order))
        elif dist == "uniform" and allow_uniform:
            rules.append(_uniform_rule(rv.order if rv.order else order))
        else:
            raise ValueError(f"Unsupported quadrature distribution: {rv.distribution}")

    points = []
    for axes in itertools.product(*[list(zip(r[0], r[1])) for r in rules]):
        xi = [float(a[0]) for a in axes]
        weight = float(np.prod([a[1] for a in axes]))
        points.append((xi, weight))
    return points

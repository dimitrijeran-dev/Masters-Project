from __future__ import annotations

import numpy as np


def weighted_moments(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    mean = float(np.sum(w * v))
    var = float(np.sum(w * (v - mean) ** 2))
    return {
        "mean": mean,
        "variance": var,
        "std": float(np.sqrt(max(var, 0.0))),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }

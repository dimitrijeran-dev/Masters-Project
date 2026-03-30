#!/usr/bin/env python3
from __future__ import annotations

import itertools


def total_order_multiindex(dim: int, order: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for alpha in itertools.product(range(order + 1), repeat=dim):
        if sum(alpha) <= order:
            out.append(tuple(int(v) for v in alpha))
    return out

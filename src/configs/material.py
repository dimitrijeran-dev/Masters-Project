from __future__ import annotations

from typing import Any, Dict


def material_payload(
    *,
    E: float,
    nu: float,
    plane_stress: bool,
    thickness: float | None = None,
    E_mean: float | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "E": float(E),
        "nu": float(nu),
        "plane_stress": bool(plane_stress),
    }
    if thickness is not None:
        payload["thickness"] = float(thickness)
    if E_mean is not None:
        payload["E_mean"] = float(E_mean)
    return payload

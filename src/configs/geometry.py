from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def resolve_crack_geometry(
    geometry_type: str,
    W: float,
    a: float,
    hole_radius: Optional[float] = None,
    hole_center: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Return canonical crack descriptors for downstream stages."""
    if geometry_type == "plate_edge_crack":
        crack_start = (0.0, 0.0)
        tip = (a, 0.0)
    elif geometry_type == "plate_hole_edge_crack":
        cx, cy = hole_center if hole_center is not None else (W / 2.0, 0.0)
        r = float(hole_radius if hole_radius is not None else 0.0)
        crack_start = (cx + r, cy)
        tip = (cx + r + a, cy)
    else:
        raise ValueError(f"Unsupported geometry_type={geometry_type!r}")

    return {
        "geometry_type": geometry_type,
        "tip": [float(tip[0]), float(tip[1])],
        "crack_start": [float(crack_start[0]), float(crack_start[1])],
        "crack_dir": [1.0, 0.0],
    }


def geometry_payload(
    *,
    geometry_type: str = "plate_edge_crack",
    W: float,
    H: float,
    a: float,
    crack_gap: float,
    hole_radius: Optional[float] = None,
    hole_center: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "geometry_type": geometry_type,
        "W": float(W),
        "H": float(H),
        "a": float(a),
        "crack_gap": float(crack_gap),
    }
    if hole_radius is not None:
        payload["hole_radius"] = float(hole_radius)
    if hole_center is not None:
        payload["hole_center"] = [float(hole_center[0]), float(hole_center[1])]

    payload.update(
        resolve_crack_geometry(
            geometry_type=geometry_type,
            W=float(W),
            a=float(a),
            hole_radius=hole_radius,
            hole_center=hole_center,
        )
    )
    return payload

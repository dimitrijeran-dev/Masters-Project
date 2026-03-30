#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RandomInput:
    name: str
    distribution: str = "normal"
    mean: float = 0.0
    std: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    order: int = 3


@dataclass
class CollocationPoint:
    index: int
    xi: List[float]
    weight: float
    mapped_parameters: Dict[str, float]


@dataclass
class QoIResult:
    case_id: str
    collocation_index: int
    xi: List[float]
    weight: float
    KI_Jstar: Optional[float]
    KI_DCM: Optional[float]
    run_dir: str
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GPCConfig:
    run_name: str
    base_output_dir: Path = Path("Data/New Data")
    run_dir: Optional[Path] = None
    random_inputs: List[RandomInput] = field(default_factory=list)
    quadrature_order: int = 3
    polynomial_order: int = 2
    solver_run_dir: Optional[Path] = None
    realization_id_start: int = 0
    compute_jstar: bool = True
    compute_dcm: bool = True
    dcm_window: Optional[Dict[str, float]] = None
    include_uniform_tensor_rule: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GPCConfig":
        random_inputs = [RandomInput(**ri) for ri in payload.get("random_inputs", [])]
        base_output = Path(payload.get("base_output_dir", "Data/New Data"))
        run_dir = payload.get("run_dir")
        solver_run_dir = payload.get("solver_run_dir")
        return cls(
            run_name=payload["run_name"],
            base_output_dir=base_output,
            run_dir=Path(run_dir) if run_dir else None,
            random_inputs=random_inputs,
            quadrature_order=int(payload.get("quadrature_order", payload.get("order", 3))),
            polynomial_order=int(payload.get("polynomial_order", 2)),
            solver_run_dir=Path(solver_run_dir) if solver_run_dir else None,
            realization_id_start=int(payload.get("realization_id_start", 0)),
            compute_jstar=bool(payload.get("compute_jstar", True)),
            compute_dcm=bool(payload.get("compute_dcm", True)),
            dcm_window=payload.get("dcm_window"),
            include_uniform_tensor_rule=bool(payload.get("include_uniform_tensor_rule", True)),
            extra=payload.get("extra", {}),
        )

    def resolved_run_dir(self) -> Path:
        return self.run_dir or (self.base_output_dir / self.run_name)

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def _load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_ki_from_validation_summary(run_dir: Path, method: str) -> Optional[float]:
    summary_path = Path(run_dir) / "validation_summary.json"
    if not summary_path.exists():
        summaries = sorted(Path(run_dir).glob("validation_summary_mc*.json"))
        if not summaries:
            return None
        summary_path = summaries[0]
    data = _load_summary(summary_path)
    m = method.lower()
    if m == "jstar":
        return data.get("KI_ref")
    if m == "dcm":
        if isinstance(data.get("dcm"), dict):
            return data["dcm"].get("KI_ref")
        return data.get("KI_ref_dcm")
    raise ValueError(f"Unsupported method: {method}")


def extract_jstar_ki(run_dir: Path) -> Optional[float]:
    return extract_ki_from_validation_summary(run_dir, "jstar")


def extract_dcm_ki(run_dir: Path) -> Optional[float]:
    return extract_ki_from_validation_summary(run_dir, "dcm")


def extract_both_qois(run_dir: Path) -> dict:
    return {
        "KI_Jstar": extract_jstar_ki(run_dir),
        "KI_DCM": extract_dcm_ki(run_dir),
    }

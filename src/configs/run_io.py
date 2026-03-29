from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def load_runtime_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "run": {},
            "geometry": {},
            "material": {},
            "stages": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("run", {})
    data.setdefault("geometry", {})
    data.setdefault("material", {})
    data.setdefault("stages", {})
    return data


def save_runtime_config(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = _jsonable(payload)
    data["schema_version"] = SCHEMA_VERSION
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return data


def update_runtime_config(path: Path, stage: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    data = load_runtime_config(path)

    for key in ("run", "geometry", "material"):
        if key in updates and isinstance(updates[key], dict):
            data.setdefault(key, {})
            data[key].update(_jsonable(updates[key]))

    data.setdefault("stages", {})
    stage_payload = _jsonable(updates.get("stage", {}))
    stage_payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    data["stages"][stage] = stage_payload

    return save_runtime_config(path, data)

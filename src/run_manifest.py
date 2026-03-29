#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def canonical_json_bytes(data: Mapping[str, Any]) -> bytes:
    payload = _to_jsonable(dict(data))
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_manifest_hash(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def write_run_manifest(run_dir: Path, manifest: Mapping[str, Any], filename: str = "run_manifest.json") -> tuple[Path, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / filename

    clean_manifest = _to_jsonable(dict(manifest))
    clean_manifest.setdefault("schema_version", "run_manifest/v1")
    clean_manifest["written_at_utc"] = datetime.now(timezone.utc).isoformat()
    clean_manifest_hash = compute_manifest_hash(clean_manifest)
    clean_manifest["manifest_hash_sha256"] = clean_manifest_hash

    manifest_path.write_text(json.dumps(clean_manifest, indent=2), encoding="utf-8")
    return manifest_path, clean_manifest_hash


def load_run_manifest(run_dir: Path, filename: str = "run_manifest.json") -> dict[str, Any]:
    manifest_path = run_dir / filename
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected = manifest.get("manifest_hash_sha256")
    if expected:
        recompute_basis = dict(manifest)
        recompute_basis.pop("manifest_hash_sha256", None)
        actual = compute_manifest_hash(recompute_basis)
        if actual != expected:
            raise ValueError(
                f"Manifest hash mismatch for {manifest_path}: expected {expected}, computed {actual}"
            )
    return manifest


def ensure_summary_manifest_hash(summary: dict[str, Any], manifest_hash: Optional[str]) -> dict[str, Any]:
    if manifest_hash:
        summary["manifest_hash_sha256"] = manifest_hash
    return summary

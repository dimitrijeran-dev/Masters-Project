from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class StudyExecutionError(RuntimeError):
    pass


class MetadataCompatibilityError(StudyExecutionError):
    pass


@dataclass
class MetadataRecord:
    step_name: str
    path: Path
    payload: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise StudyExecutionError(f"JSON file must contain an object: {path}")
    return data


def _load_metadata_records(step: Dict[str, Any]) -> List[MetadataRecord]:
    step_name = step["name"]
    records: List[MetadataRecord] = []

    metadata_path = step.get("metadata_path")
    if metadata_path:
        path = Path(metadata_path)
        if not path.exists():
            raise StudyExecutionError(f"Step '{step_name}' metadata file not found: {path}")
        records.append(MetadataRecord(step_name=step_name, path=path, payload=_read_json(path)))

    metadata_glob = step.get("metadata_glob")
    if metadata_glob:
        matches = sorted(Path().glob(metadata_glob))
        if not matches:
            raise StudyExecutionError(
                f"Step '{step_name}' metadata_glob matched no files: {metadata_glob}"
            )
        for path in matches:
            records.append(MetadataRecord(step_name=step_name, path=path, payload=_read_json(path)))

    return records


def _assert_compatible(
    baseline: MetadataRecord,
    candidate: MetadataRecord,
    keys: Iterable[str],
) -> None:
    incompatible: List[str] = []
    for key in keys:
        if key not in baseline.payload or key not in candidate.payload:
            continue
        if baseline.payload[key] != candidate.payload[key]:
            incompatible.append(
                f"{key}: baseline={baseline.payload[key]!r} ({baseline.path}) vs "
                f"candidate={candidate.payload[key]!r} ({candidate.path})"
            )

    if incompatible:
        raise MetadataCompatibilityError(
            "Incompatible metadata detected between steps "
            f"'{baseline.step_name}' and '{candidate.step_name}':\n- "
            + "\n- ".join(incompatible)
        )


def _execute_step(step: Dict[str, Any]) -> Dict[str, Any]:
    module_name = step["module"]
    function_name = step.get("function", "main")
    kwargs = step.get("kwargs", {})

    module = importlib.import_module(module_name)
    fn = getattr(module, function_name, None)
    if fn is None:
        raise StudyExecutionError(f"Function '{function_name}' not found in module '{module_name}'")
    if not callable(fn):
        raise StudyExecutionError(f"Target '{module_name}.{function_name}' is not callable")

    result = fn(**kwargs)
    if result is None:
        result = {}
    if not isinstance(result, dict):
        result = {"return_value": repr(result)}

    return result


def _collect_key_metrics(step: Dict[str, Any], records: List[MetadataRecord]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metric_specs = step.get("key_metrics", [])
    for spec in metric_specs:
        metric_name = spec["name"]
        key = spec["key"]
        source = spec.get("source", "metadata")
        if source != "metadata":
            raise StudyExecutionError(
                f"Step '{step['name']}' uses unsupported metric source '{source}'"
            )

        values = [record.payload.get(key) for record in records if key in record.payload]
        if not values:
            continue
        metrics[metric_name] = values[0] if len(values) == 1 else values

    return metrics


def run_study(manifest_path: Path, default_summary_name: str) -> Path:
    manifest = _read_json(manifest_path)
    steps = manifest.get("steps")
    if not isinstance(steps, list) or not steps:
        raise StudyExecutionError("Manifest must include a non-empty 'steps' list")

    compatibility_keys = manifest.get(
        "metadata_compatibility_keys", ["run_name", "a", "E", "nu", "plane_stress"]
    )

    summary_path = Path(
        manifest.get("summary_path", manifest_path.with_name(default_summary_name))
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_record: Optional[MetadataRecord] = None
    step_summaries: List[Dict[str, Any]] = []
    overall_artifacts: List[str] = []
    overall_metrics: Dict[str, Any] = {}

    for index, step in enumerate(steps, start=1):
        step_name = step.get("name", f"step_{index}")
        step["name"] = step_name

        step_result = _execute_step(step)
        records = _load_metadata_records(step)

        for record in records:
            if baseline_record is None:
                baseline_record = record
            else:
                _assert_compatible(baseline_record, record, compatibility_keys)

        artifacts = [str(Path(p)) for p in step.get("artifacts", [])]
        artifacts.extend(step_result.get("artifacts", []))
        metrics = _collect_key_metrics(step, records)
        metrics.update(step_result.get("key_metrics", {}))

        overall_artifacts.extend(artifacts)
        overall_metrics.update(metrics)

        step_summaries.append(
            {
                "name": step_name,
                "module": step["module"],
                "function": step.get("function", "main"),
                "metadata_files": [str(record.path) for record in records],
                "artifacts": artifacts,
                "key_metrics": metrics,
                "result": {k: v for k, v in step_result.items() if k not in {"artifacts", "key_metrics"}},
            }
        )

    summary = {
        "manifest": str(manifest_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "artifacts": sorted(set(overall_artifacts)),
        "key_metrics": overall_metrics,
        "steps": step_summaries,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary_path


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("manifest", type=Path, help="Path to study manifest JSON")
    return parser

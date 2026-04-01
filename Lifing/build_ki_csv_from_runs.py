#!/usr/bin/env python3
import argparse
import json
import csv
from pathlib import Path


def resolve_summary_path(run_dir: Path, summary_mode: str, realization_id: int | None) -> Path:
    if summary_mode == "deterministic":
        return run_dir / "validation_summary.json"
    if summary_mode == "stochastic_mean":
        return run_dir / "validation_summary_all_realizations.json"
    if summary_mode == "stochastic_realization":
        if realization_id is None:
            raise ValueError(
                "--realization-id is required when --summary-mode stochastic_realization"
            )
        return run_dir / f"validation_summary_mc{realization_id:04d}.json"
    raise ValueError(f"Unsupported summary mode: {summary_mode}")


def extract_ki_value(data: dict, summary_mode: str, ki_field: str | None, realization_id: int | None) -> float:
    if ki_field is not None:
        if ki_field not in data:
            raise KeyError(f"Requested KI field '{ki_field}' not found in summary.")
        return float(data[ki_field])

    if summary_mode == "deterministic":
        if "KI_ref" not in data:
            raise KeyError("Expected 'KI_ref' in deterministic validation summary.")
        return float(data["KI_ref"])

    if summary_mode == "stochastic_mean":
        if "KI_ref_mean" in data:
            return float(data["KI_ref_mean"])

        # Fallback: compute mean from embedded per-realization summaries if present
        realizations = data.get("summaries") or data.get("realizations")
        if realizations:
            vals = [float(r["KI_ref"]) for r in realizations if r.get("KI_ref") is not None]
            if vals:
                return float(sum(vals) / len(vals))

        raise KeyError(
            "Could not find stochastic mean KI. Expected 'KI_ref_mean' or embedded realizations/summaries."
        )

    if summary_mode == "stochastic_realization":
        if "KI_ref" not in data:
            raise KeyError(
                f"Expected 'KI_ref' in stochastic realization summary for realization_id={realization_id}."
            )
        return float(data["KI_ref"])

    raise ValueError(f"Unsupported summary mode: {summary_mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--a-values", nargs="+", type=float, required=True)
    parser.add_argument("--out-csv", required=True)

    parser.add_argument(
        "--summary-mode",
        choices=["deterministic", "stochastic_mean", "stochastic_realization"],
        default="deterministic",
        help=(
            "Which validation summary type to read. "
            "'deterministic' -> validation_summary.json, "
            "'stochastic_mean' -> validation_summary_all_realizations.json, "
            "'stochastic_realization' -> validation_summary_mcXXXX.json"
        ),
    )
    parser.add_argument(
        "--ki-field",
        default=None,
        help=(
            "Optional override for which KI field to extract from the JSON. "
            "Examples: KI_ref, KI_ref_mean, KI_ref_dcm"
        ),
    )
    parser.add_argument(
        "--realization-id",
        type=int,
        default=None,
        help="Required when --summary-mode stochastic_realization",
    )

    args = parser.parse_args()

    if len(args.run_dirs) != len(args.a_values):
        raise ValueError(
            "Length mismatch: --run-dirs and --a-values must contain the same "
            f"number of entries (got {len(args.run_dirs)} run dirs and "
            f"{len(args.a_values)} a-values)."
        )

    run_dirs = [Path(run_dir) for run_dir in args.run_dirs]
    summary_paths = [
        resolve_summary_path(run_dir, args.summary_mode, args.realization_id)
        for run_dir in run_dirs
    ]

    missing_files = [str(path) for path in summary_paths if not path.is_file()]
    if missing_files:
        missing_list = "\n  - ".join(missing_files)
        raise FileNotFoundError(
            "Missing required validation summary files:\n"
            f"  - {missing_list}"
        )

    rows = []

    for path, a in zip(summary_paths, args.a_values):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        ki_val = extract_ki_value(
            data=data,
            summary_mode=args.summary_mode,
            ki_field=args.ki_field,
            realization_id=args.realization_id,
        )

        rows.append({
            "a": float(a*1e3),  # convert m -> mm
            "KI": float(ki_val),
        })

    rows.sort(key=lambda r: r["a"])

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["a", "KI"])
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote", out_path)


if __name__ == "__main__":
    main()
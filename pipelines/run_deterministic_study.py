#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines._study_runner import build_parser, run_study


def main() -> None:
    parser = build_parser("Run a deterministic study pipeline from a manifest")
    args = parser.parse_args()
    summary_path = run_study(args.manifest, default_summary_name="deterministic_study_summary.json")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()

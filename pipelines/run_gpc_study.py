#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import json

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UQ.gpc_types import GPCConfig
from UQ.study_runner import run_gpc_study


def main() -> None:
    p = argparse.ArgumentParser(description="Run non-intrusive GPC/UQ study")
    p.add_argument("manifest", type=Path)
    args = p.parse_args()

    with args.manifest.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cfg = GPCConfig.from_dict(payload)
    out = run_gpc_study(cfg)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

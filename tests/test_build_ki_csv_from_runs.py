import csv
import json
import subprocess
import sys
from pathlib import Path


def _write_summary(run_dir: Path, ki_ref: float):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"KI_ref": ki_ref}, f)


def test_build_ki_csv_parser_and_output_sorting(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write_summary(run_a, 11.0)
    _write_summary(run_b, 9.0)

    out_csv = tmp_path / "ki_vs_a.csv"

    cmd = [
        sys.executable,
        "Lifing/build_ki_csv_from_runs.py",
        "--run-dirs",
        str(run_a),
        str(run_b),
        "--a-values",
        "0.02",
        "0.01",
        "--out-csv",
        str(out_csv),
    ]
    subprocess.run(cmd, check=True)

    with out_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert [float(r["a"]) for r in rows] == [0.01, 0.02]
    assert [float(r["KI"]) for r in rows] == [9.0, 11.0]
    assert rows[0].keys() == {"a", "KI"}

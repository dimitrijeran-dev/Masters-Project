import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from pathlib import Path

from UQ.qoi_extractors import extract_both_qois


def test_extract_both_qois_reads_validation_summary(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = {
        "KI_ref": 12.3,
        "dcm": {"KI_ref": 11.8},
    }
    (run_dir / "validation_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    out = extract_both_qois(run_dir)
    assert out["KI_Jstar"] == 12.3
    assert out["KI_DCM"] == 11.8

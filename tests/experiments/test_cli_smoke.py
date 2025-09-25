from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from datajax.frame.frame import Frame
from datajax.ir.serialize import trace_to_list

ROOT = Path(__file__).resolve().parents[2]


def _run_cli(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not existing else f"{ROOT}:{existing}"
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_export_wavespec_cli(tmp_path: Path) -> None:
    out = tmp_path / "wavespec.json"
    args = [
        "-m",
        "datajax.cli.export_wavespec",
        "--logs",
        str(ROOT / "tests" / "assets" / "sample_logs.csv"),
        "--format",
        "csv",
        "--key",
        "key",
        "--size",
        "size",
        "--out",
        str(out),
    ]
    _run_cli(args, cwd=ROOT)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["pack_order"]
    assert data["meta"]["row_count"] == 4


def test_replay_tuner_cli(tmp_path: Path) -> None:
    sample = pd.DataFrame(
        {
            "user_id": [1, 2, 1, 3],
            "unit_price": [10.0, 5.0, 2.0, 4.0],
            "qty": [2, 3, 5, 1],
        }
    )
    frame = Frame.from_pandas(sample)
    revenue = (frame.unit_price * frame.qty).rename("revenue")
    aggregated = revenue.groupby(frame.user_id).sum()
    trace = trace_to_list(aggregated.trace)
    trace_path = tmp_path / "trace.json"
    sample_path = tmp_path / "sample.parquet"
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    sample.to_parquet(sample_path)

    out = tmp_path / "policy.json"
    args = [
        "-m",
        "datajax.cli.replay_tuner",
        "--trace",
        str(trace_path),
        "--sample",
        str(sample_path),
        "--sample-format",
        "parquet",
        "--out",
        str(out),
    ]
    _run_cli(args, cwd=ROOT)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "metrics" in data and "policy" in data
    assert data["policy"]["BM"] > 0

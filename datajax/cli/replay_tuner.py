"""Offline trace replay + policy suggestion CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from datajax.api.sharding import Resource
from datajax.planner.metrics import metrics_to_dict
from datajax.planner.replay import replay_and_tune


def _load_table(path: Path, fmt: str | None = None) -> pd.DataFrame:
    if fmt is None:
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            fmt = "parquet"
        elif suffix in {".json", ".jsonl"}:
            fmt = "json"
        else:
            fmt = "csv"
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "json":
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    return pd.read_csv(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="Serialized trace JSON (list of steps)",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        required=True,
        help="Sample input table (csv/json/parquet)",
    )
    parser.add_argument(
        "--sample-format",
        choices=["csv", "json", "parquet"],
        default=None,
    )
    parser.add_argument("--backend", default="pandas", help="Planner backend name")
    parser.add_argument("--mode", default="stub", help="Planner backend mode")
    parser.add_argument(
        "--mesh-axes",
        default=None,
        help="Comma-separated mesh axes (e.g. rows,cols)",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="World size for Resource mesh",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path for metrics + policy",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    trace_data = json.loads(args.trace.read_text(encoding="utf-8"))
    sample_df = _load_table(args.sample, args.sample_format)

    resources = None
    if args.mesh_axes:
        axes = tuple(axis.strip() for axis in args.mesh_axes.split(",") if axis.strip())
        resources = Resource(mesh_axes=axes, world_size=int(args.world_size))
    elif args.world_size != 1:
        resources = Resource(mesh_axes=("rows",), world_size=int(args.world_size))

    metrics, policy = replay_and_tune(
        trace_data,
        input_df=sample_df,
        resources=resources,
        backend=args.backend,
        mode=args.mode,
    )

    payload: dict[str, Any] = {
        "metrics": metrics_to_dict(metrics),
        "policy": asdict(policy),
    }
    if isinstance(payload["policy"].get("notes"), tuple):
        payload["policy"]["notes"] = list(payload["policy"]["notes"])

    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

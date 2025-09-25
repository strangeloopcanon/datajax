"""CLI for exporting WaveSpec-style JSON from access logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from datajax.io.exporter import export_wave_spec


def _load_logs(path: Path, fmt: str | None) -> pd.DataFrame:
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


def _load_usage(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):  # pragma: no cover - user error
        raise ValueError("usage JSON must contain a dictionary mapping column -> count")
    return {str(k): int(v) for k, v in data.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs",
        type=Path,
        required=True,
        help="Path to logs file (csv/json/parquet)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "parquet"],
        help="Force input format",
        default=None,
    )
    parser.add_argument("--key", required=True, help="Column containing keys")
    parser.add_argument("--size", help="Optional column containing value sizes")
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=4,
        help="Hex prefix length for cohorts",
    )
    parser.add_argument(
        "--page-bits",
        type=int,
        default=20,
        help="Page size bits (default 1MiB)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Limit number of cohorts (None for all)",
    )
    parser.add_argument(
        "--usage-json",
        type=Path,
        help="Optional JSON mapping column -> usage count",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path for WaveSpec",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logs = _load_logs(args.logs, args.format)
    usage_counts = _load_usage(args.usage_json)

    spec = export_wave_spec(
        logs,
        key_col=args.key,
        size_col=args.size,
        prefix_len=args.prefix_len,
        page_bits=args.page_bits,
        top_k_prefixes=args.top_k,
        usage_counts=usage_counts,
    )

    args.out.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Offline exporters for cohorts, KV page extents, and WaveSpec hints.

This module provides utilities to analyze access logs or key streams and emit:
  - prefix cohorts (hot/cold groupings by key prefix),
  - coalesced KV page extents for prefetch/packing,
  - suggested pack order derived from column/key usage,
  - a WaveSpec-like structure consumable by BCache/hotweights tooling.

Where possible, it exposes zero-copy column views via DLPack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping
else:
    Iterable = Any
    Mapping = Any


@dataclass(frozen=True)
class PrefixCohort:
    prefix: str
    count: int
    bytes: int | None = None


@dataclass(frozen=True)
class KVExtent:
    start_page: int
    npages: int


def build_prefix_cohorts(
    logs: pd.DataFrame,
    *,
    key_col: str,
    size_col: str | None = None,
    prefix_len: int = 2,
    top_k: int | None = 64,
) -> list[PrefixCohort]:
    """Return aggregated cohorts by hexadecimal key prefix.

    The key column is hashed to a hex string and truncated to `prefix_len`.
    If `size_col` is provided, the per-key sizes are summed to provide bytes.
    """

    if key_col not in logs.columns:
        raise KeyError(f"Missing key column {key_col!r}")
    hashed = pd.util.hash_pandas_object(logs[key_col], index=False).astype("uint64")
    prefixes = hashed.apply(lambda x: f"{x:016x}"[: max(1, prefix_len)])
    df = pd.DataFrame({"_prefix": prefixes})
    if size_col is not None and size_col in logs.columns:
        df["_bytes"] = logs[size_col].astype("int64")
    grouped = df.groupby("_prefix")
    counts = grouped.size().rename("count").to_frame()
    if "_bytes" in df.columns:
        counts["bytes"] = grouped["_bytes"].sum()
    counts = counts.sort_values(["count"], ascending=False)
    if top_k is not None:
        counts = counts.head(int(top_k))
    out: list[PrefixCohort] = []
    for _idx, row in counts.reset_index().iterrows():
        out.append(
            PrefixCohort(
                prefix=str(row["_prefix"]),
                count=int(row["count"]),
                bytes=int(row["bytes"]) if "bytes" in counts.columns else None,
            )
        )
    return out


def coalesce_kv_extents(
    logs: pd.DataFrame | Iterable[Any],
    *,
    key_col: str | None = None,
    page_bits: int = 20,
) -> list[KVExtent]:
    """Coalesce hashed-key pages into contiguous extents.

    Parameters
    ----------
    logs:
        A DataFrame containing a key column, or an iterable of keys.
    key_col:
        Column name containing keys when logs is a DataFrame.
    page_bits:
        Number of bits representing a page (e.g., 20 bits → 1 MiB pages).
    """

    if isinstance(logs, pd.DataFrame):
        if not key_col:
            raise ValueError("key_col is required when logs is a DataFrame")
        keys = logs[key_col]
    else:
        keys = pd.Series(list(logs))

    hashed = (
        pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    )
    pages = (hashed >> 0) >> max(0, int(page_bits))
    if pages.size == 0:
        return []

    pages.sort()
    pages = np.unique(pages)

    # Coalesce consecutive pages
    extents: list[KVExtent] = []
    start = int(pages[0])
    prev = start
    for v in pages[1:]:
        iv = int(v)
        if iv == prev + 1:
            prev = iv
            continue
        extents.append(KVExtent(start_page=start, npages=prev - start + 1))
        start = iv
        prev = iv
    extents.append(KVExtent(start_page=start, npages=prev - start + 1))
    return extents


def suggest_pack_order_from_usage(usage_counts: Mapping[str, int]) -> list[str]:
    """Suggest a pack order of columns/keys by descending usage frequency."""

    return [k for k, _ in sorted(usage_counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def to_dlpack_columns(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """Expose zero-copy views for selected columns via DLPack when available.

    Returns a mapping from column name to a DLPack capsule (or numpy array as
    a fallback when __dlpack__ is not available).
    """

    out: dict[str, Any] = {}
    for c in columns:
        arr = df[c].to_numpy()
        capsule = getattr(arr, "__dlpack__", None)
        if callable(capsule):
            try:
                out[c] = capsule()  # type: ignore[no-any-return]
                continue
            except Exception:
                pass
        out[c] = arr
    return out


def export_wave_spec(
    logs: pd.DataFrame,
    *,
    key_col: str,
    size_col: str | None = None,
    prefix_len: int = 2,
    page_bits: int = 20,
    top_k_prefixes: int | None = 64,
    usage_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build a WaveSpec-like dictionary from access logs.

    The structure is intentionally simple and JSON-serializable so downstream
    tools can consume it without importing this package.
    """

    cohorts = build_prefix_cohorts(
        logs,
        key_col=key_col,
        size_col=size_col,
        prefix_len=prefix_len,
        top_k=top_k_prefixes,
    )
    extents = coalesce_kv_extents(logs, key_col=key_col, page_bits=page_bits)
    if usage_counts is None:
        # Default: assume only key usage when no counts provided
        usage_counts = {key_col: len(logs)}
    pack_order = suggest_pack_order_from_usage(usage_counts)

    approx_bytes = None
    if size_col and size_col in logs.columns:
        try:
            approx_bytes = int(logs[size_col].astype("int64").sum())
        except Exception:
            approx_bytes = None

    return {
        "pack_order": pack_order,
        "cohorts": [c.__dict__ for c in cohorts],
        "extents": [e.__dict__ for e in extents],
        "meta": {
            "page_bits": int(page_bits),
            "row_count": int(len(logs)),
            "approx_bytes": approx_bytes,
        },
    }


__all__ = [
    "PrefixCohort",
    "KVExtent",
    "build_prefix_cohorts",
    "coalesce_kv_extents",
    "suggest_pack_order_from_usage",
    "to_dlpack_columns",
    "export_wave_spec",
]

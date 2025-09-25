from __future__ import annotations

import numpy as np
import pandas as pd
from datajax.io.exporter import (
    build_prefix_cohorts,
    coalesce_kv_extents,
    export_wave_spec,
    suggest_pack_order_from_usage,
    to_dlpack_columns,
)


def test_build_prefix_cohorts_counts_sum():
    df = pd.DataFrame({"key": ["a", "a", "b", "c"], "size": [1, 2, 3, 4]})
    cohorts = build_prefix_cohorts(
        df,
        key_col="key",
        size_col="size",
        prefix_len=16,
        top_k=None,
    )
    assert sum(c.count for c in cohorts) == len(df)
    assert sum(c.bytes for c in cohorts if c.bytes is not None) == df["size"].sum()
    top = max(cohorts, key=lambda c: c.count)
    assert top.count == 2


def test_coalesce_extents_cover_unique_pages():
    df = pd.DataFrame({"key": [0, 1, 2, 10, 11]})
    extents = coalesce_kv_extents(df, key_col="key", page_bits=0)
    hashed = pd.util.hash_pandas_object(df["key"], index=False).to_numpy(
        dtype=np.uint64
    )
    expected_pages = np.unique(hashed)
    covered: list[int] = []
    for extent in extents:
        covered.extend(range(extent.start_page, extent.start_page + extent.npages))
    assert sorted(covered) == expected_pages.tolist()


def test_pack_order_and_wave_spec(tmp_path):
    df = pd.DataFrame({"key": ["u1", "u2", "u1"], "size": [10, 20, 30]})
    usage = {"colA": 5, "colB": 2}
    order = suggest_pack_order_from_usage(usage)
    assert order == ["colA", "colB"]

    spec = export_wave_spec(
        df,
        key_col="key",
        size_col="size",
        prefix_len=16,
        page_bits=0,
        top_k_prefixes=None,
        usage_counts=usage,
    )
    assert spec["pack_order"] == order
    assert spec["meta"]["row_count"] == len(df)
    assert spec["meta"]["approx_bytes"] == df["size"].sum()
    assert spec["cohorts"]
    assert spec["extents"]


def test_to_dlpack_columns_returns_capsules_or_arrays():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    outputs = to_dlpack_columns(df, ["a", "b"])
    assert set(outputs) == {"a", "b"}
    for value in outputs.values():
        if isinstance(value, np.ndarray):
            continue
        assert getattr(value, "__class__", None).__name__ == "PyCapsule"


def test_export_wave_spec_reads_csv(tmp_path):
    csv_path = tmp_path / "logs.csv"
    csv_path.write_text("key,size\na,1\nb,2\n", encoding="utf-8")
    df = pd.read_csv(csv_path)
    spec = export_wave_spec(df, key_col="key", size_col="size")
    assert spec["meta"]["row_count"] == 2
    assert spec["meta"]["approx_bytes"] == 3

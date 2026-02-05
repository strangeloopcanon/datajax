from __future__ import annotations

import pandas as pd
import pytest

from datajax import Frame, djit
from datajax.runtime import executor as runtime_executor


def _sort_for_compare(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return frame.sort_values(by, na_position="last").reset_index(drop=True)


def _reset_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_executor.reset_backend()
    monkeypatch.delenv("DATAJAX_EXECUTOR", raising=False)
    monkeypatch.delenv("DATAJAX_NATIVE_BODO", raising=False)


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_join_parity_matches_pandas_for_all_join_modes(
    how: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_runtime(monkeypatch)

    left = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "qty": [1, 2, 3, 4],
            "value": [10, 20, 30, 40],
        }
    )
    right = pd.DataFrame(
        {
            "user_id": [1, 2, 2, 5],
            "value": [100, 200, 250, 500],
            "country_code": [11, 22, 23, 55],
        }
    )

    @djit
    def pipeline(frame: Frame) -> Frame:
        return frame.join(right, on="user_id", how=how, suffixes=("_l", "_r"))

    actual = pipeline(left).to_pandas()
    expected = left.merge(right, on="user_id", how=how, suffixes=("_l", "_r"))

    pd.testing.assert_frame_equal(
        _sort_for_compare(actual, ["user_id", "country_code"]),
        _sort_for_compare(expected, ["user_id", "country_code"]),
        check_dtype=False,
    )

    record = pipeline.last_execution
    assert record is not None
    if hasattr(record.plan, "final_schema"):
        assert record.plan.final_schema == tuple(expected.columns)


def test_multikey_join_parity_matches_pandas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_runtime(monkeypatch)

    left = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "region": [1, 2, 1, 1],
            "qty": [1, 2, 3, 4],
            "value": [10, 20, 30, 40],
        }
    )
    right = pd.DataFrame(
        {
            "uid": [1, 2, 2, 4],
            "region": [1, 1, 2, 1],
            "value": [100, 200, 250, 400],
            "segment": [9, 8, 7, 6],
        }
    )

    @djit
    def pipeline(frame: Frame) -> Frame:
        return frame.join(
            right,
            left_on=["user_id", "region"],
            right_on=["uid", "region"],
            how="outer",
            suffixes=("_l", "_r"),
        )

    actual = pipeline(left).to_pandas()
    expected = left.merge(
        right,
        left_on=["user_id", "region"],
        right_on=["uid", "region"],
        how="outer",
        suffixes=("_l", "_r"),
    )

    pd.testing.assert_frame_equal(
        _sort_for_compare(actual, ["region", "user_id", "uid"]),
        _sort_for_compare(expected, ["region", "user_id", "uid"]),
        check_dtype=False,
    )

    record = pipeline.last_execution
    assert record is not None
    if hasattr(record.plan, "final_schema"):
        assert record.plan.final_schema == tuple(expected.columns)


def test_relational_filter_and_aggregate_parity(
    sample_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_runtime(monkeypatch)

    @djit
    def pipeline(frame: Frame) -> Frame:
        filtered = frame.filter(frame.qty > 1)
        total = (filtered.unit_price * filtered.qty).rename("revenue")
        return total.groupby(filtered.user_id).sum()

    actual = pipeline(sample_frame).to_pandas()
    expected_filtered = sample_frame[sample_frame["qty"] > 1]
    expected = (
        (expected_filtered["unit_price"] * expected_filtered["qty"])
        .rename("revenue")
        .groupby(expected_filtered["user_id"])
        .sum()
        .reset_index(name="revenue")
    )

    pd.testing.assert_frame_equal(
        _sort_for_compare(actual, ["user_id"]),
        _sort_for_compare(expected, ["user_id"]),
    )


def test_join_suffix_collision_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_runtime(monkeypatch)
    left = pd.DataFrame({"id": [1], "value": [10], "value_right": [11]})
    right = pd.DataFrame({"id": [1], "value": [20]})

    @djit
    def pipeline(frame: Frame) -> Frame:
        return frame.join(right, on="id", suffixes=("_right", "_right"))

    with pytest.raises(ValueError, match="collide"):
        _ = pipeline(left)

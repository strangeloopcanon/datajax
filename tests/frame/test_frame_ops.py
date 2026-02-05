from __future__ import annotations

import pandas as pd
import pytest

from datajax.api.sharding import shard
from datajax.frame.frame import Frame
from datajax.ir.graph import (
    AggregateStep,
    FilterStep,
    JoinStep,
    MapStep,
    ProjectStep,
    RepartitionStep,
)


def test_frame_filter_and_map(sample_frame: pd.DataFrame) -> None:
    frame = Frame.from_pandas(sample_frame)
    high_qty = frame.qty > 2
    filtered = frame.filter(high_qty)
    result = (filtered.unit_price * filtered.qty).rename("total")
    aggregated = result.groupby(filtered.user_id).sum()

    trace = aggregated.trace
    assert any(isinstance(step, FilterStep) for step in trace)
    assert sum(isinstance(step, MapStep) for step in trace) >= 1

    output = aggregated.to_pandas().sort_values("user_id").reset_index(drop=True)
    pandas_filtered = sample_frame[sample_frame["qty"] > 2]
    expected = (pandas_filtered["unit_price"] * pandas_filtered["qty"]).rename("total")
    expected = (
        expected.groupby(pandas_filtered["user_id"]).sum().reset_index(name="total")
    )
    expected = expected.sort_values("user_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(output, expected)


def test_frame_select_and_join(sample_frame: pd.DataFrame) -> None:
    frame = Frame.from_pandas(sample_frame)
    left = frame.select(["user_id", "qty"])
    right_df = pd.DataFrame({"user_id": [1, 2], "country": ["US", "CA"]})
    joined = left.join(right_df, on="user_id")

    trace = joined.trace
    assert any(isinstance(step, ProjectStep) for step in trace)
    assert any(isinstance(step, JoinStep) for step in trace)

    expected = sample_frame[["user_id", "qty"]].merge(right_df, on="user_id")
    output = joined.to_pandas().sort_values(["user_id", "qty"]).reset_index(drop=True)
    expected = expected.sort_values(["user_id", "qty"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(output, expected)


def test_frame_join_suffixes_and_sharding_contract(sample_frame: pd.DataFrame) -> None:
    spec = shard.by_key("user_id")
    left = Frame.from_pandas(sample_frame).repartition(spec).select(["user_id", "qty"])
    right_df = pd.DataFrame(
        {"user_id": [1, 2], "qty": [100, 200], "country": ["US", "CA"]}
    )

    joined = left.join(right_df, on="user_id", suffixes=("_left", "_right"))
    expected = left.to_pandas().merge(
        right_df,
        on="user_id",
        suffixes=("_left", "_right"),
    )
    pd.testing.assert_frame_equal(
        joined.to_pandas().sort_values(["user_id", "qty_left"]).reset_index(drop=True),
        expected.sort_values(["user_id", "qty_left"]).reset_index(drop=True),
    )
    assert joined.sharding == spec
    join_step = next(step for step in joined.trace if isinstance(step, JoinStep))
    assert join_step.suffixes == ("_left", "_right")

    right_on_qty = pd.DataFrame({"qty": [1, 2, 5], "segment": ["A", "B", "C"]})
    joined_mismatch = left.join(right_on_qty, on="qty")
    assert joined_mismatch.sharding is None

    joined_right = left.join(right_df, on="user_id", how="right")
    assert joined_right.sharding is None


def test_frame_join_left_on_right_on_multi_key_parity() -> None:
    left_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "region": [1, 2, 1, 1],
            "qty": [1, 2, 3, 4],
            "value": [10, 20, 30, 40],
        }
    )
    right_df = pd.DataFrame(
        {
            "uid": [1, 2, 2, 4],
            "region": [1, 1, 2, 1],
            "value": [100, 200, 250, 400],
            "segment": [9, 8, 7, 6],
        }
    )
    frame = Frame.from_pandas(left_df).repartition(shard.by_key("user_id"))
    joined = frame.join(
        right_df,
        left_on=("user_id", "region"),
        right_on=("uid", "region"),
        how="outer",
        suffixes=("_l", "_r"),
    )

    expected = left_df.merge(
        right_df,
        left_on=["user_id", "region"],
        right_on=["uid", "region"],
        how="outer",
        suffixes=("_l", "_r"),
    )
    output = (
        joined.to_pandas()
        .sort_values(["region", "user_id", "uid"])
        .reset_index(drop=True)
    )
    expected = expected.sort_values(["region", "user_id", "uid"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(output, expected, check_dtype=False)

    join_step = next(step for step in joined.trace if isinstance(step, JoinStep))
    assert join_step.left_on == ("user_id", "region")
    assert join_step.right_on == ("uid", "region")
    assert joined.sharding is None


def test_frame_join_rejects_suffix_collisions() -> None:
    left_df = pd.DataFrame({"id": [1], "value": [10], "value_right": [11]})
    right_df = pd.DataFrame({"id": [1], "value": [20]})
    frame = Frame.from_pandas(left_df)

    with pytest.raises(ValueError, match="collide"):
        _ = frame.join(right_df, on="id", suffixes=("_right", "_right"))


def test_frame_repartition_records_sharding(sample_frame: pd.DataFrame) -> None:
    frame = Frame.from_pandas(sample_frame)
    spec = shard.by_key("user_id")
    repartitioned = frame.repartition(spec)
    assert repartitioned.sharding == spec
    assert any(isinstance(step, RepartitionStep) for step in repartitioned.trace)


def test_groupby_additional_reductions(sample_frame: pd.DataFrame) -> None:
    frame = Frame.from_pandas(sample_frame)
    totals = (frame.unit_price * frame.qty).rename("total")
    grouped = totals.groupby(frame.user_id)

    mean_frame = grouped.mean()
    count_frame = grouped.count(alias="orders")

    pandas_totals = (sample_frame["unit_price"] * sample_frame["qty"]).rename("total")
    pandas_grouped = pandas_totals.groupby(sample_frame["user_id"])  # type: ignore[arg-type]

    expected_mean = (
        pandas_grouped.mean()
        .reset_index(name="total")
        .sort_values("user_id")
        .reset_index(drop=True)
    )
    expected_count = (
        pandas_grouped.count()
        .reset_index(name="orders")
        .sort_values("user_id")
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        mean_frame.to_pandas().sort_values("user_id").reset_index(drop=True),
        expected_mean,
    )
    pd.testing.assert_frame_equal(
        count_frame.to_pandas().sort_values("user_id").reset_index(drop=True),
        expected_count,
    )

    assert any(isinstance(step, AggregateStep) for step in mean_frame.trace)
    assert any(isinstance(step, AggregateStep) for step in count_frame.trace)

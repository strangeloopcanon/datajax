from __future__ import annotations

import pandas as pd
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

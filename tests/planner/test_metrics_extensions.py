from __future__ import annotations

from datajax.api.sharding import Resource, shard
from datajax.frame.frame import Frame
from datajax.planner.metrics import (
    estimate_plan_metrics,
    merge_runtime_counters,
    metrics_to_dict,
)
from datajax.planner.plan import build_plan


def _build_pipeline(frame: Frame) -> Frame:
    spec = shard.by_key("user_id")
    repart = frame.repartition(spec)
    filtered = repart.filter(repart.qty > 2)
    revenue = (filtered.unit_price * filtered.qty).rename("revenue")
    return revenue.groupby(filtered.user_id).sum()


def _build_pipeline_with_extra_repartition(frame: Frame) -> Frame:
    spec_key = shard.by_key("user_id")
    spec_rep = shard.replicated()
    repart = frame.repartition(spec_key)
    repart = repart.repartition(spec_rep)
    filtered = repart.filter(repart.qty > 2)
    revenue = (filtered.unit_price * filtered.qty).rename("revenue")
    return revenue.groupby(filtered.user_id).sum()


def test_build_plan_attaches_metrics(sample_frame):
    frame = Frame.from_pandas(sample_frame)
    result = _build_pipeline(frame)
    plan = build_plan(
        result.trace,
        backend="pandas",
        mode="stub",
        input_df=sample_frame,
    )
    metrics = plan.metrics
    assert metrics is not None
    assert metrics.repartition_steps == 1
    assert metrics.transform_steps >= 2
    assert metrics.aggregate_steps >= 1
    assert metrics.pack_order_hint

    metrics_dict = metrics_to_dict(metrics)
    assert metrics_dict is not None
    assert isinstance(metrics_dict["pack_order_hint"], list)


def test_shuffle_estimate_increases_with_repartition(sample_frame):
    frame1 = Frame.from_pandas(sample_frame)
    plan1 = build_plan(
        _build_pipeline(frame1).trace,
        backend="pandas",
        mode="stub",
        input_df=sample_frame,
        resources=Resource(mesh_axes=("rows",), world_size=4),
    )
    frame2 = Frame.from_pandas(sample_frame)
    plan2 = build_plan(
        _build_pipeline_with_extra_repartition(frame2).trace,
        backend="pandas",
        mode="stub",
        input_df=sample_frame,
        resources=Resource(mesh_axes=("rows",), world_size=4),
    )
    assert plan1.metrics is not None and plan2.metrics is not None
    assert plan2.metrics.repartition_steps > plan1.metrics.repartition_steps
    shuffle1 = plan1.metrics.estimated_shuffle_bytes or 0
    shuffle2 = plan2.metrics.estimated_shuffle_bytes or 0
    assert shuffle2 >= shuffle1
    assert "joins" not in plan1.metrics.notes


def test_estimate_plan_metrics_with_manual_plan(sample_frame):
    frame = Frame.from_pandas(sample_frame)
    result = _build_pipeline(frame)
    plan = build_plan(
        result.trace,
        backend="pandas",
        mode="stub",
        input_df=sample_frame,
    )
    mock_plan = type(
        "MockPlan",
        (),
        {"stages": plan.stages, "trace": plan.trace, "resources": plan.resources},
    )
    metrics = estimate_plan_metrics(mock_plan, sample_df=sample_frame)
    assert metrics.transform_steps == plan.metrics.transform_steps
    assert metrics.column_usage_counts


def test_merge_runtime_counters_overrides_fields(sample_frame):
    frame = Frame.from_pandas(sample_frame)
    plan = build_plan(
        _build_pipeline(frame).trace,
        backend="pandas",
        mode="stub",
        input_df=sample_frame,
    )
    metrics = plan.metrics
    assert metrics is not None
    counters = {
        "bytes_moved": 1234,
        "shuffle_bytes": 5678,
        "wgmma_occupancy": 0.9,
        "l2_reuse": 0.5,
        "notes": ["runtime"]
    }
    merge_runtime_counters(metrics, counters)
    assert metrics.runtime_input_bytes == 1234
    assert metrics.runtime_shuffle_bytes == 5678
    assert metrics.runtime_wgmma_occupancy == 0.9
    assert metrics.runtime_l2_reuse == 0.5
    assert metrics.runtime_notes == ["runtime"]

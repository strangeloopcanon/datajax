from __future__ import annotations

from datajax.api.sharding import Resource, shard
from datajax.frame.frame import Frame
from datajax.ir.serialize import trace_to_list
from datajax.planner.metrics import PlanMetrics
from datajax.planner.replay import replay_and_tune


def _build_trace(sample_frame):
    frame = Frame.from_pandas(sample_frame)
    spec = shard.by_key("user_id")
    repart = frame.repartition(spec)
    filtered = repart.filter(repart.qty > 2)
    revenue = (filtered.unit_price * filtered.qty).rename("revenue")
    aggregated = revenue.groupby(filtered.user_id).sum()
    return aggregated.trace


def test_replay_and_tune_from_serialized_trace(sample_frame):
    trace = _build_trace(sample_frame)
    serialized = trace_to_list(trace)
    metrics, policy = replay_and_tune(
        serialized,
        input_df=sample_frame,
        resources=Resource(mesh_axes=("rows",), world_size=4),
    )
    assert metrics.transform_steps >= 1
    assert policy.BM > 0 and policy.BN > 0 and policy.BK > 0
    assert policy.stage_depth >= 1


def test_replay_and_tune_fallback_metrics_uses_sample_df(sample_frame, monkeypatch):
    trace = _build_trace(sample_frame)
    observed: dict[str, object] = {}

    class PlanWithoutMetrics:
        def __init__(self, trace_steps):
            self.trace = tuple(trace_steps)
            self.stages = ()

    def fake_build_plan(*_args, **_kwargs):
        return PlanWithoutMetrics(trace)

    def fake_estimate(plan, *, sample_df=None):
        observed["plan"] = plan
        observed["sample_df"] = sample_df
        return PlanMetrics(
            steps_total=1,
            transform_steps=1,
            join_steps=0,
            aggregate_steps=0,
            repartition_steps=0,
            notes=["fallback"],
        )

    monkeypatch.setattr("datajax.planner.replay.build_plan", fake_build_plan)
    monkeypatch.setattr("datajax.planner.replay.estimate_plan_metrics", fake_estimate)

    replay_and_tune(trace, input_df=sample_frame)
    assert observed["sample_df"] is sample_frame

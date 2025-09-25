from __future__ import annotations

from datajax.api.sharding import Resource, shard
from datajax.frame.frame import Frame
from datajax.ir.serialize import trace_to_list
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

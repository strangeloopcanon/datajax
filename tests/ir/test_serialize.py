from __future__ import annotations

import pandas as pd
from datajax.api.sharding import shard
from datajax.ir.graph import (
    BinaryExpr,
    ColumnRef,
    InputStep,
    JoinStep,
    Literal,
    MapStep,
    RepartitionStep,
)
from datajax.ir.serialize import (
    expr_from_dict,
    expr_to_dict,
    trace_from_list,
    trace_to_list,
)


def _build_trace():
    rhs = pd.DataFrame({"user_id": [1, 2], "region": ["US", "CA"]})
    expr = BinaryExpr("mul", ColumnRef("qty"), Literal(2))
    steps = [
        InputStep(("user_id", "qty")),
        MapStep(output="qty_scaled", expr=expr),
        JoinStep(
            left_on="user_id",
            right_on="user_id",
            how="left",
            right_columns=tuple(rhs.columns),
            right_data=rhs,
        ),
        RepartitionStep(spec=shard.by_key("user_id")),
    ]
    return steps, rhs


def test_expr_roundtrip_binary():
    expr = BinaryExpr("mul", ColumnRef("x"), Literal(5))
    data = expr_to_dict(expr)
    restored = expr_from_dict(data)
    assert isinstance(restored, BinaryExpr)
    assert restored.op == "mul"


def test_trace_roundtrip_including_join_rhs():
    trace, rhs = _build_trace()
    serialized = trace_to_list(trace)
    # Tag the join RHS so we can reattach the DataFrame on load
    for item in serialized:
        if item.get("type") == "join":
            item["rhs_tag"] = "dim"
    restored = trace_from_list(serialized, rhs_tables={"dim": rhs})
    assert len(restored) == len(trace)
    assert isinstance(restored[0], InputStep)
    join_step = next(step for step in restored if isinstance(step, JoinStep))
    assert join_step.right_data is rhs


def test_trace_roundtrip_keeps_repartition_spec():
    trace, rhs = _build_trace()
    serialized = trace_to_list(trace)
    restored = trace_from_list(serialized, rhs_tables={"dim": rhs})
    repart = next(step for step in restored if isinstance(step, RepartitionStep))
    spec = repart.spec
    if isinstance(spec, dict):
        assert spec["key"] == "user_id"
    else:
        assert getattr(spec, "key", None) == "user_id"

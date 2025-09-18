from __future__ import annotations

import pandas as pd
import pytest
from datajax.frame.frame import Frame
from datajax.runtime.bodo_plan import DataJAXPlan


def _collect_lazy_nodes(plan):
    from bodo.pandas.plan import LazyPlan

    nodes = []
    stack = [plan]
    while stack:
        node = stack.pop()
        nodes.append(node)
        for arg in getattr(node, "args", []):
            if isinstance(arg, LazyPlan):
                stack.append(arg)
    return nodes


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_native_plan_join_structure(sample_frame: pd.DataFrame) -> None:
    pytest.importorskip("bodo.pandas.plan")
    import bodo
    from bodo.pandas.plan import LogicalComparisonJoin

    bodo.dataframe_library_run_parallel = False

    frame = Frame.from_pandas(sample_frame)
    left = frame.select(["user_id", "qty"])
    right_df = pd.DataFrame({"user_id": [1, 3], "country": ["US", "UK"]})
    joined = left.join(right_df, on="user_id")

    plan = DataJAXPlan(joined.trace, sample_frame)

    nodes = _collect_lazy_nodes(plan)
    assert any(isinstance(node, LogicalComparisonJoin) for node in nodes)
    assert tuple(plan.empty_data.columns) == ("user_id", "qty", "country")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_native_plan_count_dtype(sample_frame: pd.DataFrame) -> None:
    pytest.importorskip("bodo.pandas.plan")
    import bodo
    from bodo.pandas.plan import LogicalAggregate

    bodo.dataframe_library_run_parallel = False

    frame = Frame.from_pandas(sample_frame)
    totals = (frame.unit_price * frame.qty).rename("total")
    count_frame = totals.groupby(frame.user_id).count(alias="orders")

    plan = DataJAXPlan(count_frame.trace, sample_frame)
    aggregates = [
        node
        for node in _collect_lazy_nodes(plan)
        if isinstance(node, LogicalAggregate)
        or getattr(node, "plan_class", None) == "LogicalAggregate"
    ]
    assert aggregates, "Expected an aggregate node in the native plan"

    aggregate_node = aggregates[0]
    agg_exprs = aggregate_node.args[2]
    assert agg_exprs and agg_exprs[0].args[1] == "count"

    expected_dtype = pd.Series([], dtype="int64").dtype
    assert plan.empty_data["orders"].dtype == expected_dtype

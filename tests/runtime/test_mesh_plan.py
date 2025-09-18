from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from datajax.api.sharding import Resource, shard
from datajax.runtime.mesh import compute_destinations_for_mesh, mesh_shape_from_resource


def test_mesh_shape_inference_two_axes() -> None:
    mesh = Resource(mesh_axes=("rows", "cols"), world_size=8)
    shape = mesh_shape_from_resource(mesh, 8)
    assert isinstance(shape, tuple) and len(shape) == 2
    assert shape[0] * shape[1] == 8


def test_destination_mapping_two_axes_primary_first(monkeypatch: Any) -> None:
    # Validate the mixed-radix mapping logic independent of Bodo runtime
    # by calling the helper through a small shim (we don't execute rebalance).
    df = pd.DataFrame({"k": list(range(6))})

    # Monkeypatch distributed_api.rebalance to capture dests
    captured = {}

    def fake_rebalance(df, dests=None, **kwargs):
        captured["dests"] = np.asarray(dests)
        return df
    monkeypatch.setattr(
        "bodo.libs.distributed_api.rebalance",
        fake_rebalance,
        raising=False,
    )

    # Monkeypatch hashing to identity to make expectations deterministic
    monkeypatch.setattr(
        "pandas.util.hash_pandas_object",
        lambda s, index=False: s.astype("uint64"),
    )

    # 2x2 mesh, axis 0 primary: expect row-major mapping by key value
    hashed = pd.util.hash_pandas_object(df["k"], index=False).to_numpy("uint64")
    dests = compute_destinations_for_mesh(hashed, (2, 2), 0).tolist()
    # For keys 0..5: rows alternate every key, columns advance every two keys
    assert dests[:6] == [0, 2, 1, 3, 0, 2]


def test_destination_mapping_two_axes_primary_second(monkeypatch: Any) -> None:
    df = pd.DataFrame({"k": list(range(6))})
    captured = {}

    def fake_rebalance(df, dests=None, **kwargs):
        captured["dests"] = np.asarray(dests)
        return df
    monkeypatch.setattr(
        "bodo.libs.distributed_api.rebalance",
        fake_rebalance,
        raising=False,
    )

    # Identity hash for determinism
    monkeypatch.setattr(
        "pandas.util.hash_pandas_object",
        lambda s, index=False: s.astype("uint64"),
    )

    # 2x2 mesh, axis 1 primary: columns alternate first
    hashed = pd.util.hash_pandas_object(df["k"], index=False).to_numpy("uint64")
    dests = compute_destinations_for_mesh(hashed, (2, 2), 1).tolist()
    assert dests[:6] == [0, 1, 2, 3, 0, 1]


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


def test_join_inserts_rhs_rebalance_with_mesh(sample_frame):
    import bodo
    from datajax.frame.frame import Frame
    from datajax.runtime.bodo_plan import DataJAXPlan

    bodo.dataframe_library_run_parallel = False
    mesh = Resource(mesh_axes=("rows", "cols"), world_size=4)

    frame = Frame.from_pandas(sample_frame)
    left = frame.select(["user_id", "qty"]).repartition(
        shard.by_key("user_id", axis="rows")
    )
    right_df = pd.DataFrame({"user_id": [1, 3], "country": ["US", "UK"]})
    joined = left.join(right_df, on="user_id")

    plan = DataJAXPlan(joined.trace, sample_frame, resources=mesh)
    nodes = _collect_lazy_nodes(plan)
    from bodo.pandas.plan import LogicalComparisonJoin, LogicalProjection

    joins = [
        n
        for n in nodes
        if isinstance(n, LogicalComparisonJoin)
        or getattr(n, "plan_class", None) == "LogicalComparisonJoin"
    ]
    assert joins, "expected a join in the plan"
    join = joins[0]
    rhs = join.args[1]
    assert isinstance(
        rhs,
        LogicalProjection,
    ), "RHS should be wrapped in a projection for rebalance"
    # Optional: check diagnostic note exists
    if hasattr(plan, "describe"):
        notes = plan.describe()
        assert any("join: rhs_rebalanced" in n or "repartition:" in n for n in notes)

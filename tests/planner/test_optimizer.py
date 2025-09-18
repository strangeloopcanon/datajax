from __future__ import annotations

import pytest
from datajax.api.sharding import Resource, shard
from datajax.ir.graph import (
    BinaryExpr,
    ColumnRef,
    ComparisonExpr,
    FilterStep,
    InputStep,
    Literal,
    LogicalExpr,
    MapStep,
    ProjectStep,
    RepartitionStep,
)
from datajax.planner.optimizer import optimize_trace
from datajax.planner.plan import build_plan


def _dummy_trace(*steps):
    return [InputStep(("user_id", "qty", "unit_price")), *steps]


def test_filter_pushdown_moves_before_independent_map():
    trace = _dummy_trace(
        MapStep(
            output="revenue",
            expr=BinaryExpr("mul", ColumnRef("unit_price"), ColumnRef("qty")),
        ),
        FilterStep(
            predicate=ComparisonExpr("gt", ColumnRef("qty"), Literal(2)),
        ),
    )

    optimized = optimize_trace(trace)
    kinds = [type(step) for step in optimized]
    assert kinds == [InputStep, FilterStep, MapStep]


def test_filter_is_not_pushed_past_dependent_map():
    trace = _dummy_trace(
        MapStep(
            output="revenue",
            expr=BinaryExpr("mul", ColumnRef("unit_price"), ColumnRef("qty")),
        ),
        FilterStep(
            predicate=ComparisonExpr("gt", ColumnRef("revenue"), Literal(20)),
        ),
    )

    optimized = optimize_trace(trace)
    kinds = [type(step) for step in optimized]
    assert kinds == [InputStep, MapStep, FilterStep]


def test_build_plan_uses_optimized_trace_order():
    trace = _dummy_trace(
        MapStep(
            output="revenue",
            expr=BinaryExpr("mul", ColumnRef("unit_price"), ColumnRef("qty")),
        ),
        FilterStep(
            predicate=ComparisonExpr("gt", ColumnRef("qty"), Literal(2)),
        ),
    )

    plan = build_plan(trace, backend="pandas", mode="stub", resources=None)
    transform_stage = next(stage for stage in plan.stages if stage.kind == "transform")
    step_types = tuple(type(step) for step in transform_stage.steps)
    assert step_types == (FilterStep, MapStep)


def test_consecutive_projects_fuse_to_last_projection():
    trace = _dummy_trace(
        ProjectStep(columns=("user_id", "qty")),
        ProjectStep(columns=("user_id",)),
    )

    optimized = optimize_trace(trace)
    projects = [step for step in optimized if isinstance(step, ProjectStep)]
    assert len(projects) == 1
    assert projects[0].columns == ("user_id",)


def test_consecutive_filters_fuse_with_logical_and():
    trace = _dummy_trace(
        FilterStep(predicate=ComparisonExpr("gt", ColumnRef("qty"), Literal(1))),
        FilterStep(predicate=ComparisonExpr("lt", ColumnRef("qty"), Literal(5))),
    )

    optimized = optimize_trace(trace)
    filters = [step for step in optimized if isinstance(step, FilterStep)]
    assert len(filters) == 1
    predicate = filters[0].predicate
    assert isinstance(predicate, LogicalExpr)
    assert predicate.op == "and"


def test_consecutive_repartitions_with_same_spec_deduplicate():
    spec = shard.by_key("user_id")
    trace = _dummy_trace(RepartitionStep(spec=spec), RepartitionStep(spec=spec))

    optimized = optimize_trace(trace)
    repartitions = [step for step in optimized if isinstance(step, RepartitionStep)]
    assert len(repartitions) == 1


def test_build_plan_validates_mesh_axis_name():
    spec = shard.by_key("user_id", axis="cols")
    trace = _dummy_trace(RepartitionStep(spec=spec))
    resources = Resource(mesh_axes=("rows",), world_size=4)

    with pytest.raises(ValueError):
        build_plan(trace, backend="pandas", mode="stub", resources=resources)


def test_build_plan_requires_resources_when_axis_set():
    spec = shard.by_key("user_id", axis="rows")
    trace = _dummy_trace(RepartitionStep(spec=spec))

    with pytest.raises(ValueError):
        build_plan(trace, backend="pandas", mode="stub", resources=None)


def test_build_plan_validates_axis_index_range():
    spec = shard.by_key("user_id", axis=1)
    trace = _dummy_trace(RepartitionStep(spec=spec))
    resources = Resource(mesh_axes=("rows",), world_size=2)

    with pytest.raises(ValueError):
        build_plan(trace, backend="pandas", mode="stub", resources=resources)

"""Trace optimization utilities for the planner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datajax.ir.graph import (
    BinaryExpr,
    ColumnRef,
    ComparisonExpr,
    Expr,
    FilterStep,
    LogicalExpr,
    MapStep,
    ProjectStep,
    RenameExpr,
    RepartitionStep,
)
from datajax.runtime.mesh import resolve_mesh_axis

if TYPE_CHECKING:
    from collections.abc import Sequence
else:
    from collections import abc as _abc

    Sequence = _abc.Sequence

def _expr_columns(expr: Expr) -> set[str]:
    """Return the set of column names referenced by an expression."""

    if isinstance(expr, ColumnRef):
        return {expr.name}
    if isinstance(expr, RenameExpr):
        return _expr_columns(expr.expr)
    if isinstance(expr, BinaryExpr):
        return _expr_columns(expr.left) | _expr_columns(expr.right)
    if isinstance(expr, ComparisonExpr):
        return _expr_columns(expr.left) | _expr_columns(expr.right)
    if isinstance(expr, LogicalExpr):
        return _expr_columns(expr.left) | _expr_columns(expr.right)
    return set()


def _dedup_consecutive_projects(trace: Sequence[object]) -> tuple[list[object], bool]:
    optimized: list[object] = []
    changed = False
    for step in trace:
        if (
            isinstance(step, ProjectStep)
            and optimized
            and isinstance(optimized[-1], ProjectStep)
        ):
            optimized[-1] = step
            changed = True
        else:
            optimized.append(step)
    return optimized, changed


def _fuse_consecutive_filters(trace: Sequence[object]) -> tuple[list[object], bool]:
    optimized: list[object] = []
    changed = False
    for step in trace:
        if (
            isinstance(step, FilterStep)
            and optimized
            and isinstance(optimized[-1], FilterStep)
        ):
            previous = optimized[-1]
            merged_predicate = LogicalExpr("and", previous.predicate, step.predicate)
            optimized[-1] = FilterStep(predicate=merged_predicate)
            changed = True
        else:
            optimized.append(step)
    return optimized, changed


def _dedup_repartitions(trace: Sequence[object]) -> tuple[list[object], bool]:
    optimized: list[object] = []
    changed = False
    for step in trace:
        if (
            isinstance(step, RepartitionStep)
            and optimized
            and isinstance(optimized[-1], RepartitionStep)
            and optimized[-1].spec == step.spec
        ):
            optimized[-1] = step
            changed = True
        else:
            optimized.append(step)
    return optimized, changed


def _pushdown_filters(trace: Sequence[object]) -> tuple[list[object], bool]:
    optimized = list(trace)
    changed = False
    for idx, step in enumerate(list(optimized)):
        if not isinstance(step, FilterStep):
            continue
        referenced = _expr_columns(step.predicate)
        cursor = idx - 1
        while cursor >= 0:
            prev = optimized[cursor]
            if isinstance(prev, MapStep):
                if prev.output in referenced:
                    break
                optimized[cursor], optimized[cursor + 1] = (
                    optimized[cursor + 1],
                    optimized[cursor],
                )
                changed = True
                cursor -= 1
                continue
            if isinstance(prev, FilterStep):
                optimized[cursor], optimized[cursor + 1] = (
                    optimized[cursor + 1],
                    optimized[cursor],
                )
                changed = True
                cursor -= 1
                continue
            break
    return optimized, changed


def optimize_trace(trace: Sequence[object]) -> list[object]:
    """Apply simple optimizer passes (fusion, pushdown) to a trace."""

    optimized = list(trace)
    changed = True
    while changed:
        changed = False
        optimized, did_change = _dedup_consecutive_projects(optimized)
        changed = changed or did_change
        optimized, did_change = _fuse_consecutive_filters(optimized)
        changed = changed or did_change
        optimized, did_change = _dedup_repartitions(optimized)
        changed = changed or did_change
        optimized, did_change = _pushdown_filters(optimized)
        changed = changed or did_change
    return optimized


def validate_mesh_axes(trace: Sequence[object], resources: Any) -> None:
    """Ensure repartition specs reference valid mesh axes for the given resources."""

    if resources is None:
        for step in trace:
            if isinstance(step, RepartitionStep):
                spec = step.spec
                axis = getattr(spec, "axis", None)
                if axis is not None:
                    raise ValueError(
                        "ShardSpec.axis requires Resource mesh information "
                        "when provided"
                    )
        return

    axes = tuple(getattr(resources, "mesh_axes", ()) or ())

    for step in trace:
        if not isinstance(step, RepartitionStep):
            continue
        spec = step.spec
        axis = getattr(spec, "axis", None)
        if axis is None:
            continue
        if isinstance(axis, str):
            if axis not in axes:
                raise ValueError(
                    f"Unknown mesh axis {axis!r}; available axes: {axes!r}"
                )
        elif isinstance(axis, int):
            if not axes:
                raise ValueError("ShardSpec.axis index requires a non-empty mesh_axes")
            if axis < 0 or axis >= len(axes):
                raise ValueError(
                    f"Axis index {axis} is out of range for mesh axes {axes!r}"
                )
        resolve_mesh_axis(axis, resources, 0)


__all__ = ["optimize_trace", "validate_mesh_axes"]

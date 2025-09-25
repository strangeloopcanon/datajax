"""Serialization helpers for DataJAX IR expressions and steps.

These functions convert IR nodes to simple dictionaries suitable for JSON
payloads, and back again. Join RHS data are not embedded by default; callers
may rebind them during deserialization using a provided table mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datajax.ir.graph import (
    AggregateStep,
    BinaryExpr,
    ColumnRef,
    ComparisonExpr,
    Expr,
    FilterStep,
    InputStep,
    JoinStep,
    Literal,
    LogicalExpr,
    MapStep,
    ProjectStep,
    RenameExpr,
    RepartitionStep,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping, Sequence

    import pandas as pd
else:  # pragma: no cover
    import types as _types
    from collections import abc as _abc

    Iterable = _abc.Iterable
    Mapping = _abc.Mapping
    Sequence = _abc.Sequence
    pd = _types.SimpleNamespace(DataFrame=object)


def expr_to_dict(expr: Expr) -> dict[str, Any]:
    if isinstance(expr, ColumnRef):
        return {"kind": "col", "name": expr.name}
    if isinstance(expr, Literal):
        return {"kind": "lit", "value": expr.value}
    if isinstance(expr, RenameExpr):
        return {"kind": "rename", "alias": expr.alias, "expr": expr_to_dict(expr.expr)}
    if isinstance(expr, BinaryExpr):
        return {
            "kind": "bin",
            "op": expr.op,
            "left": expr_to_dict(expr.left),
            "right": expr_to_dict(expr.right),
        }
    if isinstance(expr, ComparisonExpr):
        return {
            "kind": "cmp",
            "op": expr.op,
            "left": expr_to_dict(expr.left),
            "right": expr_to_dict(expr.right),
        }
    if isinstance(expr, LogicalExpr):
        return {
            "kind": "logic",
            "op": expr.op,
            "left": expr_to_dict(expr.left),
            "right": expr_to_dict(expr.right),
        }
    raise TypeError(f"Unsupported expr for serialization: {type(expr)!r}")


def expr_from_dict(data: Mapping[str, Any]) -> Expr:
    kind = data.get("kind")
    if kind == "col":
        return ColumnRef(str(data["name"]))
    if kind == "lit":
        return Literal(data.get("value"))
    if kind == "rename":
        return RenameExpr(expr_from_dict(data["expr"]), str(data["alias"]))
    if kind == "bin":
        return BinaryExpr(
            op=str(data["op"]),
            left=expr_from_dict(data["left"]),
            right=expr_from_dict(data["right"]),
        )
    if kind == "cmp":
        return ComparisonExpr(
            op=str(data["op"]),
            left=expr_from_dict(data["left"]),
            right=expr_from_dict(data["right"]),
        )
    if kind == "logic":
        return LogicalExpr(
            op=str(data["op"]),
            left=expr_from_dict(data["left"]),
            right=expr_from_dict(data["right"]),
        )
    raise ValueError(f"Unknown expr kind: {kind!r}")


def step_to_dict(step: object) -> dict[str, Any]:
    if isinstance(step, InputStep):
        return {"type": "input", "schema": list(step.schema)}
    if isinstance(step, MapStep):
        return {"type": "map", "output": step.output, "expr": expr_to_dict(step.expr)}
    if isinstance(step, FilterStep):
        return {"type": "filter", "predicate": expr_to_dict(step.predicate)}
    if isinstance(step, ProjectStep):
        return {"type": "project", "columns": list(step.columns)}
    if isinstance(step, AggregateStep):
        return {
            "type": "aggregate",
            "key": expr_to_dict(step.key),
            "key_alias": step.key_alias,
            "value": expr_to_dict(step.value),
            "value_alias": step.value_alias,
            "agg": step.agg,
        }
    if isinstance(step, JoinStep):
        # Avoid embedding RHS data by default
        return {
            "type": "join",
            "left_on": step.left_on,
            "right_on": step.right_on,
            "how": step.how,
            "right_columns": list(step.right_columns),
            # Optional binding; caller may add a tag name to look up tables
            "rhs_tag": None,
        }
    if isinstance(step, RepartitionStep):
        spec = step.spec
        payload = {
            "kind": getattr(spec, "kind", None),
            "key": getattr(spec, "key", None),
            "axis": getattr(spec, "axis", None),
        }
        return {"type": "repartition", "spec": payload}
    raise TypeError(f"Unsupported step for serialization: {type(step)!r}")


def step_from_dict(
    data: Mapping[str, Any],
    rhs_tables: Mapping[str, pd.DataFrame] | None = None,
) -> object:
    t = data.get("type")
    if t == "input":
        return InputStep(schema=tuple(data["schema"]))
    if t == "map":
        return MapStep(output=str(data["output"]), expr=expr_from_dict(data["expr"]))
    if t == "filter":
        return FilterStep(predicate=expr_from_dict(data["predicate"]))
    if t == "project":
        return ProjectStep(columns=tuple(data["columns"]))
    if t == "aggregate":
        return AggregateStep(
            key=expr_from_dict(data["key"]),
            key_alias=str(data["key_alias"]),
            value=expr_from_dict(data["value"]),
            value_alias=str(data["value_alias"]),
            agg=str(data["agg"]),
        )
    if t == "join":
        rhs_tag = data.get("rhs_tag")
        right_df = None
        if rhs_tables is not None and rhs_tag is not None:
            right_df = rhs_tables.get(str(rhs_tag))
        return JoinStep(
            left_on=str(data["left_on"]),
            right_on=str(data["right_on"]),
            how=str(data.get("how", "inner")),
            right_columns=tuple(data.get("right_columns", ())),
            right_data=right_df,
        )
    if t == "repartition":
        spec = data.get("spec", {})
        # Preserve pass-through dict when original ShardSpec type is unavailable
        return RepartitionStep(spec=spec)
    raise ValueError(f"Unknown step type: {t!r}")


def trace_to_list(trace: Sequence[object]) -> list[dict[str, Any]]:
    return [step_to_dict(s) for s in trace]


def trace_from_list(
    serialized: Iterable[Mapping[str, Any]],
    rhs_tables: Mapping[str, pd.DataFrame] | None = None,
) -> list[object]:
    return [step_from_dict(item, rhs_tables=rhs_tables) for item in serialized]


__all__ = [
    "expr_to_dict",
    "expr_from_dict",
    "step_to_dict",
    "step_from_dict",
    "trace_to_list",
    "trace_from_list",
]

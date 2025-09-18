"""Helpers for converting execution traces into Bodo-compatible callables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

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
)

if TYPE_CHECKING:
    from collections.abc import Iterable
else:
    from collections import abc as _abc

    Iterable = _abc.Iterable

_BINARY_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
}

_COMPARISON_SYMBOLS = {
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
}

_LOGICAL_SYMBOLS = {
    "and": "&",
    "or": "|",
}


def _expr_to_code(expr: Expr) -> str:
    if isinstance(expr, ColumnRef):
        return f"frame[{expr.name!r}]"
    if isinstance(expr, Literal):
        return repr(expr.value)
    if isinstance(expr, RenameExpr):
        return _expr_to_code(expr.expr)
    if isinstance(expr, BinaryExpr):
        op = _BINARY_SYMBOLS.get(expr.op)
        if op is None:
            raise ValueError(f"Unsupported binary op for Bodo codegen: {expr.op}")
        left = _expr_to_code(expr.left)
        right = _expr_to_code(expr.right)
        return f"({left}) {op} ({right})"
    if isinstance(expr, ComparisonExpr):
        op = _COMPARISON_SYMBOLS.get(expr.op)
        if op is None:
            raise ValueError(f"Unsupported comparison op for Bodo codegen: {expr.op}")
        left = _expr_to_code(expr.left)
        right = _expr_to_code(expr.right)
        return f"({left}) {op} ({right})"
    if isinstance(expr, LogicalExpr):
        op = _LOGICAL_SYMBOLS.get(expr.op)
        if op is None:
            raise ValueError(f"Unsupported logical op for Bodo codegen: {expr.op}")
        left = _expr_to_code(expr.left)
        right = _expr_to_code(expr.right)
        return f"(({left}) {op} ({right}))"
    raise TypeError(f"Unsupported expression type for Bodo codegen: {type(expr)!r}")


def generate_bodo_callable(
    trace: Iterable[object],
    *,
    reuse_namespace: dict[str, object] | None = None,
):
    """Return a Python function that executes the trace using pandas ops."""

    lines: list[str] = ["def _datajax_bodo_impl(frame):", "    frame = frame.copy()"]
    constants: dict[str, object] = reuse_namespace.copy() if reuse_namespace else {}
    join_index = 0

    for step in trace:
        if isinstance(step, MapStep):
            expr_code = _expr_to_code(step.expr)
            lines.append(f"    frame[{step.output!r}] = {expr_code}")
        elif isinstance(step, FilterStep):
            predicate_code = _expr_to_code(step.predicate)
            lines.append(f"    frame = frame.loc[{predicate_code}]")
        elif isinstance(step, AggregateStep):
            key = step.key_alias
            value = step.value_alias
            aggregation = (
                f"    frame = frame.groupby({key!r}, as_index=False)"
                f"[[{value!r}]].agg({{{value!r}: 'sum'}})"
            )
            lines.append(aggregation)
        elif isinstance(step, InputStep):
            continue
        elif isinstance(step, ProjectStep):
            cols = ", ".join(repr(col) for col in step.columns)
            lines.append(f"    frame = frame[[{cols}]]")
        elif isinstance(step, JoinStep):
            const_name = f"_datajax_join_rhs_{join_index}"
            join_index += 1
            constants[const_name] = step.right_data
            lines.append(
                f"    frame = frame.merge({const_name}, left_on={step.left_on!r}, "
                f"right_on={step.right_on!r}, how={step.how!r})"
            )
        else:
            raise TypeError(f"Unsupported IR step for Bodo codegen: {step!r}")

    lines.append("    return frame")
    source = "\n".join(lines)
    namespace: dict[str, object] = {"pd": pd}
    namespace.update(constants)
    exec(source, namespace, namespace)
    impl = namespace["_datajax_bodo_impl"]
    namespace.pop("_datajax_bodo_impl", None)
    return impl, source, namespace


__all__ = ["generate_bodo_callable"]

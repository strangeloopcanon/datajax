"""Frame tracer and lightweight execution helpers."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

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
    RepartitionStep,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from datajax.api.sharding import ShardSpec
else:
    from collections import abc as _abc

    Iterable = _abc.Iterable
    Sequence = _abc.Sequence
    ShardSpec = Any

SeriesLike = Union["SeriesExpr", int, float, bool]

_BINARY_OPERATORS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "truediv": operator.truediv,
}

_COMPARISON_OPERATORS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}

_LOGICAL_OPERATORS = {
    "and": operator.and_,
    "or": operator.or_,
}


def _ensure_expr(frame: Frame, value: SeriesLike) -> Expr:
    if isinstance(value, SeriesExpr):
        if value.frame is not frame:
            raise ValueError("Cannot combine SeriesExpr from different frames")
        return value.expr
    if isinstance(value, (int, float, bool)):
        return Literal(value)
    raise TypeError(f"Unsupported value for expression: {type(value)!r}")


def _infer_alias(expr: Expr, fallback: str) -> str:
    if isinstance(expr, RenameExpr):
        return expr.alias
    if isinstance(expr, ColumnRef):
        return expr.name
    return fallback


class Frame:
    """A traced dataframe wrapper backed by pandas for local execution."""

    def __init__(
        self,
        data: pd.DataFrame,
        trace: Sequence[object],
        sharding: ShardSpec | None = None,
    ) -> None:
        self._data = data
        self._trace: list[object] = list(trace)
        self.schema = tuple(data.columns)
        self._sharding = sharding

    @classmethod
    def from_pandas(
        cls,
        data: pd.DataFrame,
        *,
        sharding: ShardSpec | None = None,
    ) -> Frame:
        copied = data.copy()
        return cls(copied, trace=[InputStep(tuple(copied.columns))], sharding=sharding)

    @property
    def trace(self) -> Sequence[object]:
        return tuple(self._trace)

    @property
    def sharding(self) -> ShardSpec | None:
        return self._sharding

    def to_pandas(self) -> pd.DataFrame:
        """Return a copy of the underlying pandas DataFrame."""

        return self._data.copy()

    def __getitem__(self, column: str | SeriesExpr) -> SeriesExpr | Frame:
        if isinstance(column, str):
            if column not in self._data.columns:
                raise KeyError(column)
            return SeriesExpr(self, ColumnRef(column))
        if isinstance(column, SeriesExpr):
            # This is a simplified check. We should check the dtype of the expression.
            return self.filter(column)
        raise TypeError(f"Unsupported key type: {type(column)}")

    def __getattr__(self, name: str) -> SeriesExpr:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if name in self._data.columns:
                return SeriesExpr(self, ColumnRef(name))
            raise

    def _with_appended_steps(
        self,
        data: pd.DataFrame,
        steps: Iterable[object],
        *,
        sharding: ShardSpec | None = None,
    ) -> Frame:
        new_trace = list(self._trace)
        new_trace.extend(steps)
        sharding_value = sharding if sharding is not None else self._sharding
        return Frame(data, new_trace, sharding=sharding_value)

    def _evaluate_expr(self, expr: Expr) -> pd.Series:
        if isinstance(expr, ColumnRef):
            return self._data[expr.name]
        if isinstance(expr, Literal):
            return pd.Series([expr.value] * len(self._data), index=self._data.index)
        if isinstance(expr, BinaryExpr):
            op = _BINARY_OPERATORS.get(expr.op)
            if op is None:
                raise ValueError(f"Unsupported binary op {expr.op}")
            left = self._evaluate_expr(expr.left)
            right = self._evaluate_expr(expr.right)
            return op(left, right)
        if isinstance(expr, ComparisonExpr):
            op = _COMPARISON_OPERATORS.get(expr.op)
            if op is None:
                raise ValueError(f"Unsupported comparison op {expr.op}")
            left = self._evaluate_expr(expr.left)
            right = self._evaluate_expr(expr.right)
            return op(left, right)
        if isinstance(expr, LogicalExpr):
            op = _LOGICAL_OPERATORS.get(expr.op)
            if op is None:
                raise ValueError(f"Unsupported logical op {expr.op}")
            left = self._evaluate_expr(expr.left)
            right = self._evaluate_expr(expr.right)
            return op(left, right)
        if isinstance(expr, RenameExpr):
            base = self._evaluate_expr(expr.expr)
            return base.rename(expr.alias)
        raise TypeError(f"Unsupported expression type: {type(expr)!r}")

    def _aggregate(
        self,
        value: SeriesExpr,
        key: SeriesExpr,
        *,
        agg: str,
        value_alias: str | None = None,
    ) -> Frame:
        value_expr = _ensure_expr(self, value)
        key_expr = _ensure_expr(self, key)
        inferred_value_alias = _infer_alias(value_expr, "value")
        inferred_key_alias = _infer_alias(key_expr, "key")
        value_alias = value_alias or inferred_value_alias

        value_series = self._evaluate_expr(value_expr)
        if not value_series.name:
            value_series = value_series.rename(value_alias)
        key_series = self._evaluate_expr(key_expr)
        key_name = key_series.name or inferred_key_alias
        if not key_series.name:
            key_series = key_series.rename(key_name)

        grouped = value_series.groupby(key_series)
        if agg == "count":
            aggregated = grouped.count()
        else:
            aggregated = grouped.agg(agg)
        result_name = value_series.name or value_alias
        result = aggregated.reset_index(name=result_name)
        result = result.rename(columns={result.columns[0]: key_name})

        map_step = MapStep(output=value_alias, expr=value_expr)
        agg_step = AggregateStep(
            key=key_expr,
            key_alias=key_name,
            value=value_expr,
            value_alias=result_name,
            agg=agg,
        )
        return self._with_appended_steps(result, [map_step, agg_step])

    def aggregate_sum(self, value: SeriesExpr, key: SeriesExpr) -> Frame:
        return self._aggregate(value, key, agg="sum")

    def select(self, columns: Sequence[str]) -> Frame:
        desired = list(columns)
        missing = [col for col in desired if col not in self._data.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        projected = self._data.loc[:, desired]
        return self._with_appended_steps(
            projected,
            [ProjectStep(columns=tuple(desired))],
        )

    def join(
        self,
        other: Frame | pd.DataFrame,
        *,
        on: str,
        how: str = "inner",
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> Frame:
        if isinstance(other, Frame):
            right_df = other.to_pandas()
        elif isinstance(other, pd.DataFrame):
            right_df = other.copy()
        else:
            raise TypeError("join expects a Frame or pandas.DataFrame as the RHS")
        joined = self._data.merge(
            right_df,
            left_on=on,
            right_on=on,
            how=how,
            suffixes=suffixes,
        )
        step = JoinStep(
            left_on=on,
            right_on=on,
            how=how,
            right_columns=tuple(right_df.columns),
            right_data=right_df,
        )
        return self._with_appended_steps(joined, [step])

    def repartition(self, spec: ShardSpec) -> Frame:
        step = RepartitionStep(spec=spec)
        return self._with_appended_steps(self._data.copy(), [step], sharding=spec)

    def filter(self, predicate: SeriesExpr) -> Frame:
        pred_expr = _ensure_expr(self, predicate)
        mask = self._evaluate_expr(pred_expr)
        if mask.dtype != bool:
            raise TypeError("Filter predicate must evaluate to booleans")
        filtered = self._data.loc[mask]
        return self._with_appended_steps(filtered, [FilterStep(predicate=pred_expr)])

@dataclass(frozen=True)
class SeriesExpr:
    frame: Frame
    expr: Expr

    def rename(self, alias: str) -> SeriesExpr:
        return SeriesExpr(self.frame, RenameExpr(self.expr, alias))

    def groupby(self, key: SeriesExpr) -> GroupedSeriesExpr:
        if key.frame is not self.frame:
            raise ValueError("Groupby key must stem from the same frame")
        return GroupedSeriesExpr(frame=self.frame, value=self, key=key)

    def _binary(self, op: str, other: SeriesLike) -> SeriesExpr:
        rhs = _ensure_expr(self.frame, other)
        return SeriesExpr(self.frame, BinaryExpr(op=op, left=self.expr, right=rhs))

    def __mul__(self, other: SeriesLike) -> SeriesExpr:
        return self._binary("mul", other)

    def __rmul__(self, other: SeriesLike) -> SeriesExpr:
        return self._binary("mul", other)

    def __add__(self, other: SeriesLike) -> SeriesExpr:
        return self._binary("add", other)

    def __radd__(self, other: SeriesLike) -> SeriesExpr:
        return self._binary("add", other)

    def __sub__(self, other: SeriesLike) -> SeriesExpr:
        return self._binary("sub", other)

    def __rsub__(self, other: SeriesLike) -> SeriesExpr:
        rhs = _ensure_expr(self.frame, other)
        return SeriesExpr(self.frame, BinaryExpr(op="sub", left=rhs, right=self.expr))

    def __truediv__(self, other: SeriesLike) -> SeriesExpr:
        return self._binary("truediv", other)

    def __rtruediv__(self, other: SeriesLike) -> SeriesExpr:
        rhs = _ensure_expr(self.frame, other)
        return SeriesExpr(
            self.frame,
            BinaryExpr(op="truediv", left=rhs, right=self.expr),
        )

    def _compare(self, op: str, other: SeriesLike) -> SeriesExpr:
        rhs = _ensure_expr(self.frame, other)
        return SeriesExpr(self.frame, ComparisonExpr(op=op, left=self.expr, right=rhs))

    def __eq__(self, other: SeriesLike) -> SeriesExpr:  # type: ignore[override]
        return self._compare("eq", other)

    def __ne__(self, other: SeriesLike) -> SeriesExpr:  # type: ignore[override]
        return self._compare("ne", other)

    def __lt__(self, other: SeriesLike) -> SeriesExpr:
        return self._compare("lt", other)

    def __le__(self, other: SeriesLike) -> SeriesExpr:
        return self._compare("le", other)

    def __gt__(self, other: SeriesLike) -> SeriesExpr:
        return self._compare("gt", other)

    def __ge__(self, other: SeriesLike) -> SeriesExpr:
        return self._compare("ge", other)

    def _logical(self, op: str, other: SeriesExpr) -> SeriesExpr:
        if other.frame is not self.frame:
            raise ValueError("Logical operations require Series from the same frame")
        return SeriesExpr(
            self.frame,
            LogicalExpr(op=op, left=self.expr, right=other.expr),
        )

    def __and__(self, other: SeriesExpr) -> SeriesExpr:
        return self._logical("and", other)

    def __or__(self, other: SeriesExpr) -> SeriesExpr:
        return self._logical("or", other)


@dataclass(frozen=True)
class GroupedSeriesExpr:
    frame: Frame
    value: SeriesExpr
    key: SeriesExpr

    def sum(self) -> Frame:
        return self.frame._aggregate(self.value, self.key, agg="sum")

    def mean(self) -> Frame:
        return self.frame._aggregate(self.value, self.key, agg="mean")

    def min(self) -> Frame:
        return self.frame._aggregate(self.value, self.key, agg="min")

    def max(self) -> Frame:
        return self.frame._aggregate(self.value, self.key, agg="max")

    def count(self, *, alias: str | None = None) -> Frame:
        literal_one = SeriesExpr(self.frame, Literal(1))
        return self.frame._aggregate(
            literal_one,
            self.key,
            agg="count",
            value_alias=alias or "count",
        )

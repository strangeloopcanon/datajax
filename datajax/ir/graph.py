"""Lightweight IR primitives for DataJAX traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Expr:
    """Base class for scalar or columnar expressions."""


@dataclass(frozen=True)
class ColumnRef(Expr):
    name: str


@dataclass(frozen=True)
class Literal(Expr):
    value: Any


@dataclass(frozen=True)
class BinaryExpr(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class RenameExpr(Expr):
    expr: Expr
    alias: str


@dataclass(frozen=True)
class ComparisonExpr(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class LogicalExpr(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class InputStep:
    schema: tuple[str, ...]


@dataclass(frozen=True)
class MapStep:
    output: str
    expr: Expr


@dataclass(frozen=True)
class AggregateStep:
    key: Expr
    key_alias: str
    value: Expr
    value_alias: str
    agg: str


@dataclass(frozen=True)
class FilterStep:
    predicate: Expr


@dataclass(frozen=True)
class ProjectStep:
    columns: tuple[str, ...]


@dataclass(frozen=True)
class JoinStep:
    left_on: str
    right_on: str
    how: str
    right_columns: tuple[str, ...]
    right_data: Any


@dataclass(frozen=True)
class RepartitionStep:
    spec: Any

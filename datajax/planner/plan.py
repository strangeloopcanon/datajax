"""Execution plan representations generated from traces."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from datajax.api.sharding import join_output_sharding
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
from datajax.ir.join_semantics import build_join_column_plan
from datajax.planner.metrics import PlanMetrics, estimate_plan_metrics, metrics_to_dict
from datajax.planner.optimizer import optimize_trace, validate_mesh_axes

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import pandas as pd

    from datajax.runtime.bodo_plan import DataJAXPlan
else:
    import types as _types
    from collections import abc as _abc

    Iterable = _abc.Iterable
    Sequence = _abc.Sequence
    pd = _types.SimpleNamespace(DataFrame=object)
    DataJAXPlan = Any

try:  # pragma: no cover - optional dependency
    from datajax.runtime.bodo_plan import (
        DataJAXPlan as _RuntimeDataJAXPlan,  # type: ignore
    )
except Exception:  # pragma: no cover - optional dependency
    _RuntimeDataJAXPlan = None


@dataclass(frozen=True)
class Stage:
    kind: str
    steps: Sequence[object]
    input_schema: tuple[str, ...]
    output_schema: tuple[str, ...]
    target_sharding: object | None
    output_sharding: object | None = None

    def describe(self) -> str:
        names = ", ".join(type(step).__name__ for step in self.steps)
        return f"{self.kind}: {names}" if names else self.kind

    def explain_lines(self) -> list[str]:
        header = f"{self.kind}: {self.input_schema} -> {self.output_schema}"
        if self.target_sharding is not None:
            header += f" in_sharding={_format_sharding(self.target_sharding)}"
        if (
            self.output_sharding is not None
            and self.output_sharding != self.target_sharding
        ):
            header += f" out_sharding={_format_sharding(self.output_sharding)}"
        lines = [header]
        for step in self.steps:
            lines.append(f"  - {_format_step(step)}")
        return lines


@dataclass(frozen=True)
class ExecutionPlan:
    backend: str
    mode: str
    stages: Sequence[Stage]
    trace: Sequence[object]
    final_schema: tuple[str, ...]
    final_sharding: object | None
    resources: object | None
    metrics: PlanMetrics | None = None

    def describe(self) -> list[str]:
        return [stage.describe() for stage in self.stages]

    def explain(
        self,
        *,
        include_metrics: bool = True,
        include_trace: bool = False,
    ) -> str:
        lines: list[str] = [
            (
                f"ExecutionPlan backend={self.backend} mode={self.mode} "
                f"stages={len(self.stages)}"
            ),
            (
                f"final_schema={self.final_schema} "
                f"sharding={_format_sharding(self.final_sharding)}"
            ),
        ]

        if self.resources is not None:
            lines.append(f"resources={_format_resources(self.resources)}")

        if include_metrics and self.metrics is not None:
            lines.append("metrics:")
            lines.extend(_format_metrics(self.metrics))

        lines.append("stages:")
        for idx, stage in enumerate(self.stages):
            stage_lines = stage.explain_lines()
            if stage_lines:
                stage_lines[0] = f"[{idx}] {stage_lines[0]}"
            lines.extend(stage_lines)

        if include_trace:
            lines.append("trace:")
            for step in self.trace:
                lines.append(f"  - {_format_step(step)}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        from datajax.ir.serialize import step_to_dict

        stages = []
        for stage in self.stages:
            stages.append(
                {
                    "kind": stage.kind,
                    "input_schema": list(stage.input_schema),
                    "output_schema": list(stage.output_schema),
                    "target_sharding": _sharding_to_dict(stage.target_sharding),
                    "output_sharding": _sharding_to_dict(stage.output_sharding),
                    "steps": [step_to_dict(step) for step in stage.steps],
                }
            )
        return {
            "backend": self.backend,
            "mode": self.mode,
            "final_schema": list(self.final_schema),
            "final_sharding": _sharding_to_dict(self.final_sharding),
            "resources": _resources_to_dict(self.resources),
            "metrics": _metrics_to_dict(self.metrics),
            "stages": stages,
            "trace": [step_to_dict(step) for step in self.trace],
        }


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
    "and": "and",
    "or": "or",
}


def _format_expr(expr: Expr) -> str:
    if isinstance(expr, ColumnRef):
        return expr.name
    if isinstance(expr, Literal):
        return repr(expr.value)
    if isinstance(expr, RenameExpr):
        return f"{_format_expr(expr.expr)} as {expr.alias}"
    if isinstance(expr, BinaryExpr):
        op = _BINARY_SYMBOLS.get(expr.op, expr.op)
        return f"({_format_expr(expr.left)} {op} {_format_expr(expr.right)})"
    if isinstance(expr, ComparisonExpr):
        op = _COMPARISON_SYMBOLS.get(expr.op, expr.op)
        return f"({_format_expr(expr.left)} {op} {_format_expr(expr.right)})"
    if isinstance(expr, LogicalExpr):
        op = _LOGICAL_SYMBOLS.get(expr.op, expr.op)
        return f"({_format_expr(expr.left)} {op} {_format_expr(expr.right)})"
    return type(expr).__name__


def _sharding_to_dict(spec: object | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    if isinstance(spec, dict):
        return {str(k): v for k, v in spec.items()}
    kind = getattr(spec, "kind", None)
    if kind is not None:
        out: dict[str, Any] = {"kind": str(kind)}
        key = getattr(spec, "key", None)
        if key is not None:
            out["key"] = key
        axis = getattr(spec, "axis", None)
        if axis is not None:
            out["axis"] = axis
        return out
    return {"repr": repr(spec)}


def _format_sharding(spec: object | None) -> str:
    data = _sharding_to_dict(spec)
    if data is None:
        return "-"
    if "repr" in data and len(data) == 1:
        return str(data["repr"])
    return ", ".join(f"{k}={v!r}" for k, v in data.items())


def _resources_to_dict(resources: object | None) -> dict[str, Any] | None:
    if resources is None:
        return None
    if isinstance(resources, dict):
        return {str(k): v for k, v in resources.items()}
    out: dict[str, Any] = {"type": type(resources).__name__}
    mesh_axes = getattr(resources, "mesh_axes", None)
    if mesh_axes is not None:
        out["mesh_axes"] = list(mesh_axes)
    world_size = getattr(resources, "world_size", None)
    if world_size is not None:
        out["world_size"] = world_size
    return out


def _format_resources(resources: object) -> str:
    data = _resources_to_dict(resources)
    if data is None:
        return "-"
    if data.get("type") == "dict":
        return repr(data)
    parts = []
    for key in ("mesh_axes", "world_size"):
        if key in data:
            parts.append(f"{key}={data[key]!r}")
    rendered = ", ".join(parts) if parts else repr(resources)
    return f"{data.get('type', type(resources).__name__)}({rendered})"


def _format_step(step: object) -> str:
    if isinstance(step, InputStep):
        return f"InputStep schema={tuple(step.schema)!r}"
    if isinstance(step, MapStep):
        return f"MapStep output={step.output!r} expr={_format_expr(step.expr)}"
    if isinstance(step, FilterStep):
        return f"FilterStep predicate={_format_expr(step.predicate)}"
    if isinstance(step, ProjectStep):
        return f"ProjectStep columns={tuple(step.columns)!r}"
    if isinstance(step, AggregateStep):
        return (
            f"AggregateStep agg={step.agg!r} "
            f"key={_format_expr(step.key)} value={_format_expr(step.value)} "
            f"-> ({step.key_alias!r}, {step.value_alias!r})"
        )
    if isinstance(step, JoinStep):
        left_keys = step.left_on[0] if len(step.left_on) == 1 else step.left_on
        right_keys = step.right_on[0] if len(step.right_on) == 1 else step.right_on
        if step.suffixes != ("_x", "_y"):
            return (
                f"JoinStep how={step.how!r} on=({left_keys!r}={right_keys!r}) "
                f"rhs_cols={len(step.right_columns)} suffixes={step.suffixes!r}"
            )
        return (
            f"JoinStep how={step.how!r} on=({left_keys!r}={right_keys!r}) "
            f"rhs_cols={len(step.right_columns)}"
        )
    if isinstance(step, RepartitionStep):
        return f"RepartitionStep spec={_format_sharding(step.spec)}"
    return type(step).__name__


def _format_metrics(metrics: PlanMetrics) -> list[str]:
    lines = [
        "  "
        f"steps_total={metrics.steps_total} transform={metrics.transform_steps} "
        f"join={metrics.join_steps} aggregate={metrics.aggregate_steps} "
        f"repartition={metrics.repartition_steps}"
    ]
    if metrics.estimated_input_rows is not None:
        lines.append(f"  estimated_input_rows={metrics.estimated_input_rows}")
    if metrics.estimated_input_bytes is not None:
        lines.append(f"  estimated_input_bytes={metrics.estimated_input_bytes}")
    if metrics.estimated_shuffle_bytes is not None:
        lines.append(f"  estimated_shuffle_bytes={metrics.estimated_shuffle_bytes}")
    if metrics.runtime_input_bytes is not None:
        lines.append(f"  runtime_input_bytes={metrics.runtime_input_bytes}")
    if metrics.runtime_shuffle_bytes is not None:
        lines.append(f"  runtime_shuffle_bytes={metrics.runtime_shuffle_bytes}")
    if metrics.pack_order_hint:
        lines.append(f"  pack_order_hint={metrics.pack_order_hint}")
    if metrics.notes:
        lines.append(f"  notes={metrics.notes}")
    if metrics.runtime_notes:
        lines.append(f"  runtime_notes={metrics.runtime_notes}")
    return lines


def _metrics_to_dict(metrics: PlanMetrics | None) -> dict[str, Any] | None:
    return metrics_to_dict(metrics)


_STEP_KINDS = {
    InputStep: "input",
    MapStep: "transform",
    FilterStep: "transform",
    ProjectStep: "transform",
    JoinStep: "join",
    AggregateStep: "aggregate",
    RepartitionStep: "repartition",
}


def _classify_step(step: object) -> str:
    for step_type, kind in _STEP_KINDS.items():
        if isinstance(step, step_type):
            return kind
    return type(step).__name__.lower()


def _update_schema(schema: tuple[str, ...], step: object) -> tuple[str, ...]:
    if isinstance(step, MapStep):
        if step.output in schema:
            return schema
        return schema + (step.output,)
    if isinstance(step, ProjectStep):
        return tuple(step.columns)
    if isinstance(step, AggregateStep):
        return (step.key_alias, step.value_alias)
    if isinstance(step, JoinStep):
        plan = build_join_column_plan(
            left_columns=schema,
            right_columns=step.right_columns,
            left_on=step.left_on,
            right_on=step.right_on,
            suffixes=step.suffixes,
        )
        return plan.output_columns
    return schema


def _group_into_stages(
    trace: Iterable[object],
) -> tuple[list[Stage], tuple[str, ...], object | None]:
    stages: list[Stage] = []
    current_kind: str | None = None
    current_steps: list[object] = []
    current_schema: tuple[str, ...] = ()
    current_sharding: object | None = None
    stage_input_schema: tuple[str, ...] = ()
    stage_sharding: object | None = None

    def flush() -> None:
        nonlocal current_kind
        nonlocal current_steps
        nonlocal stage_input_schema
        nonlocal current_schema
        nonlocal stage_sharding
        if current_steps and current_kind is not None:
            stages.append(
                Stage(
                    kind=current_kind,
                    steps=tuple(current_steps),
                    input_schema=stage_input_schema,
                    output_schema=current_schema,
                    target_sharding=stage_sharding,
                    output_sharding=current_sharding,
                )
            )
        current_kind = None
        current_steps = []
        stage_input_schema = current_schema
        stage_sharding = current_sharding

    for step in trace:
        if isinstance(step, InputStep):
            if current_steps:
                flush()
            current_schema = tuple(step.schema)
            stage_input_schema = current_schema
            stage_sharding = current_sharding
            stages.append(
                Stage(
                    kind="input",
                    steps=(step,),
                    input_schema=current_schema,
                    output_schema=current_schema,
                    target_sharding=current_sharding,
                    output_sharding=current_sharding,
                )
            )
            continue

        kind = _classify_step(step)
        if current_kind is None:
            current_kind = kind
            stage_input_schema = current_schema
            stage_sharding = current_sharding
        if kind != current_kind and current_steps:
            flush()
            current_kind = kind
            stage_input_schema = current_schema
            stage_sharding = current_sharding

        current_steps.append(step)
        if isinstance(step, RepartitionStep):
            current_sharding = step.spec
        else:
            current_schema = _update_schema(current_schema, step)
            if isinstance(step, JoinStep):
                current_sharding = join_output_sharding(
                    current_sharding,
                    left_on=step.left_on,
                    how=step.how,
                )

    flush()
    return stages, current_schema, current_sharding


def _apply_step_sharding_contract(
    current: object | None, step: object
) -> object | None:
    if isinstance(step, RepartitionStep):
        return step.spec
    if isinstance(step, JoinStep):
        return join_output_sharding(current, left_on=step.left_on, how=step.how)
    return current


def validate_stage_distribution_contracts(stages: Sequence[Stage]) -> None:
    previous_output: object | None = None
    seen_input_stage = False
    for stage in stages:
        if stage.kind == "input":
            previous_output = stage.output_sharding
            seen_input_stage = True
            continue
        if not seen_input_stage:
            raise ValueError("Execution plan is missing an input stage")
        if stage.target_sharding != previous_output:
            raise ValueError(
                "Stage sharding contract mismatch: "
                f"stage={stage.kind!r} expected_input={previous_output!r} "
                f"actual_input={stage.target_sharding!r}"
            )
        expected_output = stage.target_sharding
        for step in stage.steps:
            expected_output = _apply_step_sharding_contract(expected_output, step)
        if stage.output_sharding != expected_output:
            raise ValueError(
                "Stage sharding output mismatch: "
                f"stage={stage.kind!r} expected_output={expected_output!r} "
                f"actual_output={stage.output_sharding!r}"
            )
        previous_output = stage.output_sharding


def build_plan(
    trace: Sequence[object],
    *,
    backend: str,
    mode: str,
    input_df: pd.DataFrame | None = None,
    resources: object | None = None,
) -> ExecutionPlan | DataJAXPlan:
    optimized_trace = optimize_trace(trace)
    validate_mesh_axes(optimized_trace, resources)

    stages, final_schema, final_sharding = _group_into_stages(optimized_trace)
    validate_stage_distribution_contracts(stages)

    native_flag = os.environ.get("DATAJAX_NATIVE_BODO", "0") == "1"
    if backend == "bodo" and native_flag:
        if _RuntimeDataJAXPlan is None:
            raise RuntimeError(
                "DATAJAX_NATIVE_BODO=1 requested but native plan support is unavailable"
            )
        if input_df is None:
            raise ValueError("input_df is required for native Bodo lowering")
        return _RuntimeDataJAXPlan(optimized_trace, input_df, resources=resources)

    # Pre-compute optional metrics using a lightweight plan-like object
    try:
        metrics = estimate_plan_metrics(
            SimpleNamespace(
                stages=tuple(stages),
                trace=tuple(optimized_trace),
                resources=resources,
            ),
            sample_df=input_df,
        )
    except Exception:
        metrics = None

    plan = ExecutionPlan(
        backend=backend,
        mode=mode,
        stages=tuple(stages),
        trace=tuple(optimized_trace),
        final_schema=final_schema,
        final_sharding=final_sharding,
        resources=resources,
        metrics=metrics,
    )
    return plan


__all__ = [
    "ExecutionPlan",
    "Stage",
    "build_plan",
    "validate_stage_distribution_contracts",
]

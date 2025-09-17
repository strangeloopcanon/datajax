
"""Execution plan representations generated from traces."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import pandas as pd

from datajax.ir.graph import (
    AggregateStep,
    FilterStep,
    InputStep,
    JoinStep,
    MapStep,
    ProjectStep,
    RepartitionStep,
)
try:
    from datajax.runtime.bodo_plan import DataJAXPlan  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    DataJAXPlan = None  # type: ignore


@dataclass(frozen=True)
class Stage:
    kind: str
    steps: Sequence[object]
    input_schema: Tuple[str, ...]
    output_schema: Tuple[str, ...]
    target_sharding: object | None

    def describe(self) -> str:
        names = ", ".join(type(step).__name__ for step in self.steps)
        return f"{self.kind}: {names}" if names else self.kind


@dataclass(frozen=True)
class ExecutionPlan:
    backend: str
    mode: str
    stages: Sequence[Stage]
    trace: Sequence[object]
    final_schema: Tuple[str, ...]
    final_sharding: object | None
    resources: object | None

    def describe(self) -> List[str]:
        return [stage.describe() for stage in self.stages]


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


def _update_schema(schema: Tuple[str, ...], step: object) -> Tuple[str, ...]:
    if isinstance(step, MapStep):
        if step.output in schema:
            return schema
        return schema + (step.output,)
    if isinstance(step, ProjectStep):
        return tuple(step.columns)
    if isinstance(step, AggregateStep):
        return (step.key_alias, step.value_alias)
    if isinstance(step, JoinStep):
        new_cols = list(schema)
        for col in step.right_columns:
            if col not in new_cols:
                new_cols.append(col)
        return tuple(new_cols)
    return schema


def _group_into_stages(trace: Iterable[object]) -> Tuple[List[Stage], Tuple[str, ...], object | None]:
    stages: List[Stage] = []
    current_kind: str | None = None
    current_steps: List[object] = []
    current_schema: Tuple[str, ...] = ()
    current_sharding: object | None = None
    stage_input_schema: Tuple[str, ...] = ()
    stage_sharding: object | None = None

    def flush() -> None:
        nonlocal current_kind, current_steps, stage_input_schema, current_schema, stage_sharding
        if current_steps and current_kind is not None:
            stages.append(
                Stage(
                    kind=current_kind,
                    steps=tuple(current_steps),
                    input_schema=stage_input_schema,
                    output_schema=current_schema,
                    target_sharding=stage_sharding,
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

    flush()
    return stages, current_schema, current_sharding


def build_plan(
    trace: Sequence[object],
    *,
    backend: str,
    mode: str,
    input_df: pd.DataFrame = None,
    resources: object | None = None,
) -> ExecutionPlan | DataJAXPlan:
    stages, final_schema, final_sharding = _group_into_stages(trace)

    native_flag = os.environ.get("DATAJAX_NATIVE_BODO", "0") == "1"
    if backend == "bodo" and native_flag:
        if DataJAXPlan is None:
            raise RuntimeError(
                "DATAJAX_NATIVE_BODO=1 requested but native plan support is unavailable"
            )
        if input_df is None:
            raise ValueError("input_df is required for native Bodo lowering")
        return DataJAXPlan(trace, input_df, resources=resources)

    return ExecutionPlan(
        backend=backend,
        mode=mode,
        stages=tuple(stages),
        trace=tuple(trace),
        final_schema=final_schema,
        final_sharding=final_sharding,
        resources=resources,
    )


__all__ = ["ExecutionPlan", "Stage", "build_plan"]

"""Execution helpers that run stage plans on pandas or Bodo backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datajax.runtime import bodo_codegen

if TYPE_CHECKING:
    import pandas as pd

    from datajax.planner.plan import ExecutionPlan, Stage
else:
    import types as _types

    pd = _types.SimpleNamespace(DataFrame=object)
    ExecutionPlan = Stage = Any


def _run_transform_stage(frame: pd.DataFrame, stage: Stage) -> pd.DataFrame:
    for step in stage.steps:
        step_type = type(step).__name__
        if step_type == "MapStep":
            expr_fn, _, _ = bodo_codegen.generate_bodo_callable([step])
            frame = expr_fn(frame)
        elif step_type == "FilterStep":
            expr_fn, _, _ = bodo_codegen.generate_bodo_callable([step])
            frame = expr_fn(frame)
        elif step_type == "ProjectStep":
            expr_fn, _, _ = bodo_codegen.generate_bodo_callable([step])
            frame = expr_fn(frame)
        else:
            raise NotImplementedError(f"Unsupported transform step {step_type}")
    return frame


def _run_join_stage(frame: pd.DataFrame, stage: Stage) -> pd.DataFrame:
    expr_fn, _, _ = bodo_codegen.generate_bodo_callable(stage.steps)
    return expr_fn(frame)


def _run_aggregate_stage(frame: pd.DataFrame, stage: Stage) -> pd.DataFrame:
    expr_fn, _, _ = bodo_codegen.generate_bodo_callable(stage.steps)
    return expr_fn(frame)


def execute_plan(
    plan: ExecutionPlan,
    *,
    frame: pd.DataFrame,
    backend_mode: str,
) -> pd.DataFrame:
    current = frame
    for stage in plan.stages:
        if stage.kind == "transform":
            current = _run_transform_stage(current, stage)
        elif stage.kind == "join":
            current = _run_join_stage(current, stage)
        elif stage.kind == "aggregate":
            current = _run_aggregate_stage(current, stage)
        elif stage.kind in {"input", "repartition"}:
            continue
        else:
            raise NotImplementedError(f"Unsupported stage kind {stage.kind}")
    return current


__all__ = ["execute_plan"]

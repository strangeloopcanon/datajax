"""Utilities for compiling planner stages with Bodo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from datajax.runtime import bodo_codegen

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd

    from datajax.planner.plan import ExecutionPlan, Stage
else:
    import types as _types
    from collections import abc as _abc

    Callable = _abc.Callable
    Sequence = _abc.Sequence
    pd = _types.SimpleNamespace(DataFrame=object)
    ExecutionPlan = Stage = Any


@dataclass(frozen=True)
class CompiledStage:
    stage: Stage
    fn: Callable[[pd.DataFrame], pd.DataFrame]
    source: str


@dataclass(frozen=True)
class CompiledPlan:
    stages: Sequence[CompiledStage]
    final_schema: tuple[str, ...]
    sharding: object | None
    trace_signature: tuple[object, ...]

    def run(self, frame: pd.DataFrame) -> pd.DataFrame:
        current = frame
        for compiled in self.stages:
            current = compiled.fn(current)
        return current


def compile_plan_with_backend(plan: ExecutionPlan, backend) -> CompiledPlan:
    from datajax.planner.plan import Stage

    namespace_cache: dict[str, object] | None = None
    compiled_stages: list[CompiledStage] = []

    for stage in plan.stages:
        if stage.kind == "input":
            continue
        if stage.kind in {"transform", "join", "aggregate"}:
            fn, source, namespace_cache = bodo_codegen.generate_bodo_callable(
                stage.steps, reuse_namespace=namespace_cache
            )
            compiled_fn = backend.compile_callable(fn)
            compiled_stages.append(
                CompiledStage(stage=stage, fn=compiled_fn, source=source)
            )
        elif stage.kind == "repartition":
            def identity(frame: pd.DataFrame) -> pd.DataFrame:
                return frame

            compiled_stages.append(
                CompiledStage(stage=stage, fn=identity, source="repartition")
            )
        else:
            raise NotImplementedError(f"Unsupported stage kind {stage.kind}")

    if not compiled_stages:
        fn, source, namespace_cache = bodo_codegen.generate_bodo_callable(
            plan.trace, reuse_namespace=namespace_cache
        )
        compiled_fn = backend.compile_callable(fn)
        input_schema = plan.stages[0].input_schema if plan.stages else tuple()
        fallback_stage = Stage(
            kind="fallback",
            steps=plan.trace,
            input_schema=input_schema,
            output_schema=plan.final_schema,
            target_sharding=plan.final_sharding,
        )
        compiled_stages.append(
            CompiledStage(stage=fallback_stage, fn=compiled_fn, source=source)
        )

    return CompiledPlan(
        stages=tuple(compiled_stages),
        final_schema=plan.final_schema,
        sharding=plan.final_sharding,
        trace_signature=tuple(plan.trace),
    )


__all__ = ["CompiledPlan", "CompiledStage", "compile_plan_with_backend"]

"""djit implementation with Bodo-first execution and tracing."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import wraps
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pandas as pd

from datajax.frame.frame import Frame
from datajax.planner.metrics import estimate_plan_metrics, merge_runtime_counters
from datajax.planner.plan import build_plan
from datajax.runtime.bodo_pipeline import compile_plan_with_backend
from datajax.runtime.executor import active_backend_name, execute, get_active_backend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from datajax.api.sharding import Resource
    from datajax.planner.plan import ExecutionPlan
    from datajax.runtime.bodo_pipeline import CompiledPlan
else:
    Callable = Sequence = Any
    Resource = ExecutionPlan = CompiledPlan = Any


def _wrap_arg(value: Any) -> Any:
    if isinstance(value, Frame):
        return value
    if isinstance(value, pd.DataFrame):
        return Frame.from_pandas(value)
    return value


def _unwrap_result(result: Any) -> Frame:
    if isinstance(result, Frame):
        return result
    raise TypeError("djit functions must return a Frame instance")


@dataclass
class ExecutionRecord:
    trace: Sequence[object]
    output: Frame
    backend: str
    plan: ExecutionPlan
    backend_mode: str
    sharding: object | None
    resources: Resource | None


class DjitFunction:
    """Callable wrapper that records traces from djit-compiled functions."""

    def __init__(self, fn: Callable[..., Frame]):
        self._fn = fn
        self._backend = get_active_backend()
        self._compiled: Callable[..., Any] | None = None
        self._backend_name = active_backend_name()
        self._backend_mode = getattr(self._backend, "mode", self._backend_name)
        self._last_execution: ExecutionRecord | None = None
        self._generated_source: str | None = None
        self._compiled_plan: CompiledPlan | None = None
        self._resources: Resource | None = None
        wraps(fn)(self)

    @property
    def resources(self) -> Resource | None:
        return self._resources

    @resources.setter
    def resources(self, value: Resource | None) -> None:
        self._resources = value

    def __call__(self, *args: Any, **kwargs: Any) -> Frame:
        if self._backend_name == "bodo":
            return self._call_bodo(*args, **kwargs)
        return self._call_python(*args, **kwargs)

    def lower(self, *args: Any, **kwargs: Any) -> Sequence[object]:
        """Run the function and return the resulting trace."""

        result = self(*args, **kwargs)
        return result.trace

    @property
    def last_execution(self) -> ExecutionRecord | None:
        return self._last_execution

    def _sample_dataframe(self, args: tuple[Any, ...]) -> pd.DataFrame | None:
        if not args:
            return None
        first = args[0]
        if isinstance(first, Frame):
            return first.to_pandas()
        if isinstance(first, pd.DataFrame):
            return first
        return None

    def _ensure_plan_metrics(
        self,
        plan: Any,
        *,
        sample_df: pd.DataFrame | None,
    ) -> Any:
        metrics = getattr(plan, "metrics", None)
        if metrics is None:
            proxy = SimpleNamespace(
                stages=getattr(plan, "stages", ()),
                trace=tuple(getattr(plan, "trace", ())),
                resources=getattr(plan, "resources", None),
            )
            try:
                metrics = estimate_plan_metrics(proxy, sample_df=sample_df)
            except Exception:
                metrics = None
        return metrics

    def _runtime_counters_from_env(self) -> dict[str, Any] | None:
        path = os.environ.get("DATAJAX_RUNTIME_METRICS")
        if not path:
            return None
        try:
            if path.strip().startswith("{"):
                return json.loads(path)
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def _apply_runtime_metrics(
        self,
        plan: Any,
        *,
        sample_df: pd.DataFrame | None,
    ) -> None:
        metrics = self._ensure_plan_metrics(plan, sample_df=sample_df)
        if metrics is None:
            return
        counters = self._runtime_counters_from_env()
        if not counters:
            return
        try:
            merge_runtime_counters(metrics, counters)
        except Exception:
            return

    def _call_python(self, *args: Any, **kwargs: Any) -> Frame:
        self._compiled_plan = None
        sample_df = self._sample_dataframe(args)
        wrapped_args = tuple(_wrap_arg(arg) for arg in args)
        wrapped_kwargs = {key: _wrap_arg(value) for key, value in kwargs.items()}
        if self._backend_mode in {"stub", "pandas"}:
            if self._compiled is None:
                self._compiled = self._backend.compile_callable(self._fn)
            result = _unwrap_result(self._compiled(*wrapped_args, **wrapped_kwargs))
        else:
            result = _unwrap_result(self._fn(*wrapped_args, **wrapped_kwargs))
        plan = build_plan(
            result.trace,
            backend=self._backend_name,
            mode=self._backend_mode,
            resources=self._resources,
        )
        self._apply_runtime_metrics(plan, sample_df=sample_df)
        final_sharding = getattr(plan, "final_sharding", None)
        materialized = Frame(result.to_pandas(), result.trace, final_sharding)
        self._last_execution = ExecutionRecord(
            trace=result.trace,
            output=materialized,
            backend=self._backend_name,
            plan=plan,
            backend_mode=self._backend_mode,
            sharding=final_sharding,
            resources=self._resources,
        )
        return materialized

    def _call_bodo(self, *args: Any, **kwargs: Any) -> Frame:
        wrapped_args = tuple(_wrap_arg(arg) for arg in args)
        wrapped_kwargs = {key: _wrap_arg(value) for key, value in kwargs.items()}
        traced_result = _unwrap_result(self._fn(*wrapped_args, **wrapped_kwargs))

        input_df = None
        if args and isinstance(args[0], Frame):
            input_df = args[0].to_pandas()
        elif args and isinstance(args[0], pd.DataFrame):
            input_df = args[0]

        plan = build_plan(
            traced_result.trace,
            backend=self._backend_name,
            mode=self._backend_mode,
            input_df=input_df,
            resources=self._resources,
        )

        if hasattr(plan, "plan_class"):
            result_df = execute(plan)
        else:
            if input_df is None:
                raise ValueError("Bodo execution requires a pandas DataFrame input")
            compiled = compile_plan_with_backend(plan, self._backend)
            self._compiled_plan = compiled
            result_df = compiled.run(input_df)

        # Normalize pyarrow-backed dtypes to NumPy equivalents for consistency
        result_df = result_df.copy()
        for column in result_df.columns:
            dtype = result_df[column].dtype
            if hasattr(dtype, "pyarrow_dtype"):
                to_pandas = dtype.pyarrow_dtype.to_pandas_dtype()
                result_df[column] = result_df[column].astype(to_pandas)

        final_sharding = getattr(plan, "final_sharding", None)
        self._apply_runtime_metrics(plan, sample_df=input_df)
        result_frame = Frame(result_df, traced_result.trace, final_sharding)

        self._last_execution = ExecutionRecord(
            trace=traced_result.trace,
            output=result_frame,
            backend=self._backend_name,
            plan=plan,
            backend_mode=self._backend_mode,
            sharding=final_sharding,
            resources=self._resources,
        )
        return result_frame


def djit(fn: Callable[..., Frame]) -> DjitFunction:
    """Decorator that wraps a pure function operating on Frame instances."""

    return DjitFunction(fn)

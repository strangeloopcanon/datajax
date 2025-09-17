"""djit implementation with Bodo-first execution and tracing."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from datajax.api.sharding import Resource
from datajax.frame.frame import Frame
from datajax.planner.plan import ExecutionPlan, build_plan
from datajax.runtime.bodo_pipeline import CompiledPlan, compile_plan_with_backend
from datajax.runtime.executor import execute
from datajax.runtime.executor import active_backend_name, get_active_backend, execute


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
        self._compiled: Optional[Callable[..., Any]] = None
        self._backend_name = active_backend_name()
        self._backend_mode = getattr(self._backend, "mode", self._backend_name)
        self._last_execution: Optional[ExecutionRecord] = None
        self._generated_source: Optional[str] = None
        self._compiled_plan: Optional[CompiledPlan] = None
        self._resources: Optional[Resource] = None
        wraps(fn)(self)

    @property
    def resources(self) -> Optional[Resource]:
        return self._resources

    @resources.setter
    def resources(self, value: Optional[Resource]) -> None:
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
    def last_execution(self) -> Optional[ExecutionRecord]:
        return self._last_execution

    def _call_python(self, *args: Any, **kwargs: Any) -> Frame:
        self._compiled_plan = None
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
        materialized = Frame(result.to_pandas(), result.trace, plan.final_sharding)
        self._last_execution = ExecutionRecord(
            trace=result.trace,
            output=materialized,
            backend=self._backend_name,
            plan=plan,
            backend_mode=self._backend_mode,
            sharding=plan.final_sharding,
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
                result_df[column] = result_df[column].astype(dtype.pyarrow_dtype.to_pandas_dtype())

        result_frame = Frame(result_df, traced_result.trace, plan.final_sharding)

        self._last_execution = ExecutionRecord(
            trace=traced_result.trace,
            output=result_frame,
            backend=self._backend_name,
            plan=plan,
            backend_mode=self._backend_mode,
            sharding=plan.final_sharding,
            resources=self._resources,
        )
        return result_frame


def djit(fn: Callable[..., Frame]) -> DjitFunction:
    """Decorator that wraps a pure function operating on Frame instances."""

    return DjitFunction(fn)

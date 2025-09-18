
"""Execution backend management preferring Bodo by default."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import warnings
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol

from datajax.runtime import bodo_stub

if TYPE_CHECKING:
    from collections.abc import Callable

    from datajax.planner.plan import ExecutionPlan
    from datajax.runtime.bodo_plan import DataJAXPlan
else:
    from collections import abc as _abc

    Callable = _abc.Callable
    ExecutionPlan = DataJAXPlan = Any


class ExecutionBackend(Protocol):
    name: str

    def compile_callable(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Return a callable ready for execution on this backend."""


class PandasBackend:
    name = "pandas"
    mode = "pandas"

    def compile_callable(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn


class BodoBackend:
    name = "bodo"

    def __init__(self) -> None:
        use_stub = os.environ.get("DATAJAX_USE_BODO_STUB", "1") != "0"
        if use_stub:
            self._available = True
            self._error = None
            self._jit = bodo_stub.jit
            self.mode = "stub"
            if "bodo" not in sys.modules:
                module = ModuleType("bodo")
                module.jit = bodo_stub.jit  # type: ignore[attr-defined]
                module.config = bodo_stub.config  # type: ignore[attr-defined]
                module.__version__ = bodo_stub.__version__  # type: ignore[attr-defined]
                module.__datajax_stub__ = True  # type: ignore[attr-defined]
                sys.modules["bodo"] = module
        else:
            existing = sys.modules.get("bodo")
            if existing is not None and getattr(existing, "__datajax_stub__", False):
                del sys.modules["bodo"]

            spec = importlib.util.find_spec("bodo")
            if spec is None:
                self._available = False
                self._error = ImportError("bodo module not found")
                self._jit = None
                self.mode = "missing"
            else:
                allow_import = os.environ.get("DATAJAX_ALLOW_BODO_IMPORT", "0") == "1"
                if not allow_import:
                    self._available = False
                    self._error = RuntimeError(
                        "Bodo detected but DATAJAX_ALLOW_BODO_IMPORT=1 is "
                        "required to import it"
                    )
                    self._jit = None
                    self.mode = "present_disabled"
                else:
                    bodo_module = importlib.import_module("bodo")
                    self._available = True
                    self._error = None
                    self._jit = bodo_module.jit
                    self.mode = "real"

    @property
    def available(self) -> bool:
        return self._available

    def compile_callable(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        if not self._available:
            raise RuntimeError("Bodo backend is not available") from self._error
        assert self._jit is not None
        return self._jit(fn)


@dataclass
class BackendHandle:
    backend: ExecutionBackend
    source: str


_ACTIVE_BACKEND: BackendHandle | None = None


def _choose_backend() -> BackendHandle:
    requested = os.environ.get("DATAJAX_EXECUTOR", "auto").lower()
    if requested not in {"auto", "bodo", "pandas"}:
        warnings.warn(
            f"Unknown DATAJAX_EXECUTOR={requested!r}; defaulting to auto",
            stacklevel=2,
        )
        requested = "auto"

    if requested in {"auto", "bodo"}:
        bodo_backend = BodoBackend()
        if bodo_backend.available:
            return BackendHandle(backend=bodo_backend, source=bodo_backend.mode)
        if requested == "bodo":
            raise RuntimeError("DATAJAX_EXECUTOR=bodo but Bodo is not installed")
        warnings.warn(
            "Bodo backend unavailable; falling back to pandas executor",
            category=RuntimeWarning,
            stacklevel=2,
        )

    return BackendHandle(backend=PandasBackend(), source="pandas")


def get_active_backend() -> ExecutionBackend:
    global _ACTIVE_BACKEND
    if _ACTIVE_BACKEND is None:
        _ACTIVE_BACKEND = _choose_backend()
    return _ACTIVE_BACKEND.backend


def active_backend_name() -> str:
    return get_active_backend().name


def reset_backend() -> None:
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = None


def execute(plan: ExecutionPlan | DataJAXPlan, *args, **kwargs) -> Any:
    """Executes a plan using the active backend."""
    from datajax.runtime.bodo_plan import DataJAXPlan

    if isinstance(plan, DataJAXPlan):
        from bodo.pandas.plan import execute_plan as execute_bodo_plan

        return execute_bodo_plan(plan)

    # For now, we assume that the ExecutionPlan is a single-stage plan
    # that contains a callable.
    if len(plan.stages) != 1:
        raise NotImplementedError("Multi-stage plans are not yet supported")

    stage = plan.stages[0]
    if not callable(stage.steps[0]):
        raise TypeError("Expected a callable step in the execution plan")

    fn = stage.steps[0]
    backend = get_active_backend()
    compiled_fn = backend.compile_callable(fn)
    return compiled_fn(*args, **kwargs)


__all__ = [
    "ExecutionBackend",
    "PandasBackend",
    "BodoBackend",
    "get_active_backend",
    "active_backend_name",
    "reset_backend",
    "execute",
]

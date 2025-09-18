"""Lightweight Bodo stub providing the minimal API used in tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypeVar, overload

if TYPE_CHECKING:
    from collections.abc import Callable
else:
    from collections import abc as _abc

    Callable = _abc.Callable

F = TypeVar("F", bound=Callable[..., Any])


@overload
def jit(fn: None = None, *, parallel: bool | None = None, **_: Any) -> Callable[[F], F]:
    ...


@overload
def jit(fn: F, *, parallel: bool | None = None, **_: Any) -> F:
    ...


def jit(
    fn: F | None = None,
    *,
    parallel: bool | None = None,
    **_: Any,
) -> F | Callable[[F], F]:
    """Return a decorator that behaves like bodo.jit but simply returns the input."""

    def decorator(inner: F) -> F:
        return inner

    if fn is None:
        return decorator
    return decorator(fn)


config = SimpleNamespace(num_workers=1)
__version__ = "stub"

__all__ = ["jit", "config", "__version__"]

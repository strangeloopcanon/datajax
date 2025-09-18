"""High-level transform entry points: vmap, pjit, scan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from datajax.api.djit import DjitFunction
from datajax.frame.frame import Frame

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from datajax.api.sharding import Resource, ShardSpec
    from datajax.planner.plan import ExecutionPlan
else:
    from collections import abc as _abc

    Callable = _abc.Callable
    Iterable = _abc.Iterable
    Sequence = _abc.Sequence
    Resource = ShardSpec = ExecutionPlan = Any


@dataclass
class PartitionedFunction:
    fn: Callable[..., Any]
    in_shardings: ShardSpec | None
    out_shardings: ShardSpec | None
    resources: Resource | None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Validate axis vs mesh early (before executing)
        if self.out_shardings is not None and getattr(
            self.out_shardings, "axis", None
        ) is not None:
            axis = self.out_shardings.axis
            if self.resources is None or not getattr(self.resources, "mesh_axes", None):
                raise ValueError(
                    "ShardSpec.axis was provided but no Resource mesh is configured"
                )
            axes = tuple(self.resources.mesh_axes)
            if isinstance(axis, str):
                if axis not in axes:
                    raise ValueError(
                        f"Unknown mesh axis {axis!r}; available axes: {axes!r}"
                    )
            elif isinstance(axis, int):
                if axis < 0 or axis >= len(axes):
                    raise ValueError(
                        f"Axis index {axis} is out of range for mesh axes {axes!r}"
                    )

        base = self.fn
        previous_resources = None
        if isinstance(base, DjitFunction):
            previous_resources = base.resources
            base.resources = self.resources
        try:
            result = base(*args, **kwargs)
        finally:
            if isinstance(base, DjitFunction):
                base.resources = previous_resources
        if isinstance(result, Frame) and self.out_shardings is not None:
            record = None
            if isinstance(base, DjitFunction):
                record = base.last_execution
            if record is not None and record.sharding is not None:
                if record.sharding != self.out_shardings:
                    raise ValueError(
                        f"Output sharding {record.sharding!r} does not match "
                        f"expected {self.out_shardings!r}"
                    )
        return result

    def lower(self, *args: Any, **kwargs: Any) -> Sequence[object]:
        base = self.fn
        if isinstance(base, DjitFunction):
            previous_resources = base.resources
            base.resources = self.resources
            try:
                return base.lower(*args, **kwargs)
            finally:
                base.resources = previous_resources
        if hasattr(base, "lower"):
            return base.lower(*args, **kwargs)
        raise TypeError("Wrapped function does not expose a lowering method")

    @property
    def last_plan(self) -> ExecutionPlan | None:
        base = self.fn
        if isinstance(base, DjitFunction) and base.last_execution is not None:
            return base.last_execution.plan
        return None


def pjit(
    fn: Callable[..., Any],
    *,
    in_shardings: ShardSpec | None = None,
    out_shardings: ShardSpec | None = None,
    resources: Resource | None = None,
) -> PartitionedFunction:
    """Wrap a function with sharding metadata.

    This is currently a thin wrapper that forwards execution to the underlying
    callable while preserving djit traces for offline inspection.
    """

    return PartitionedFunction(
        fn=fn,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        resources=resources,
    )


class VmapFunction:
    def __init__(self, fn: Callable[..., Any]):
        self._fn = fn

    def __call__(
        self,
        batched: Iterable[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Sequence[Any]:
        return [self._fn(item, *args, **kwargs) for item in batched]


def vmap(fn: Callable[..., Any]) -> VmapFunction:
    return VmapFunction(fn)


class ScanFunction:
    def __init__(self, fn: Callable[[Any, Any], tuple[Any, Any]], *, init_carry: Any):
        self._fn = fn
        self._carry = init_carry

    def __call__(self, sequence: Iterable[Any]) -> tuple[Any, Sequence[Any]]:
        outputs = []
        carry = self._carry
        for item in sequence:
            carry, out = self._fn(carry, item)
            outputs.append(out)
        self._carry = carry
        return carry, outputs


def scan(fn: Callable[[Any, Any], tuple[Any, Any]], *, init: Any) -> ScanFunction:
    return ScanFunction(fn, init_carry=init)


__all__ = ["pjit", "PartitionedFunction", "vmap", "scan"]

"""Sharding descriptors used by pjit-style APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Resource:
    mesh_axes: tuple[str, ...]
    world_size: int
    axis_sizes: tuple[int, ...] | None = None
    devices: tuple[int, ...] | None = None
    annotations: dict[str, str] | None = None


@dataclass(frozen=True)
class ShardSpec:
    kind: str
    key: str | None = None
    axis: str | int | None = None

    def describe(self) -> str:
        if self.kind == "key" and self.key:
            return f"key({self.key})"
        return self.kind


class ShardBuilder:
    """Factory for shard specifications."""

    def by_key(self, column: str, *, axis: str | int | None = None) -> ShardSpec:
        """Shard by a key column, optionally along a specific mesh axis.

        axis may be a mesh axis name (e.g., "rows") or an integer index.
        """
        return ShardSpec(kind="key", key=column, axis=axis)

    def replicated(self) -> ShardSpec:
        return ShardSpec(kind="replicated", key=None)


shard = ShardBuilder()


def _spec_attr(spec: object | None, name: str) -> Any:
    if spec is None:
        return None
    if isinstance(spec, dict):
        return spec.get(name)
    return getattr(spec, name, None)


def join_output_sharding(
    spec: object | None,
    *,
    left_on: str,
    how: str,
) -> object | None:
    """Return a conservative post-join sharding contract.

    Preserve key sharding only when the current key matches the left join key and
    the join does not introduce unmatched RHS rows into the output (inner/left).
    Replicated sharding remains replicated.
    """

    kind = _spec_attr(spec, "kind")
    if kind == "replicated":
        return spec
    if kind != "key":
        return None
    key = _spec_attr(spec, "key")
    if key != left_on:
        return None
    if how in {"inner", "left"}:
        return spec
    return None


__all__ = ["Resource", "ShardSpec", "join_output_sharding", "shard"]

"""Sharding descriptors used by pjit-style APIs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Resource:
    mesh_axes: tuple[str, ...]
    world_size: int


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


__all__ = ["Resource", "ShardSpec", "shard"]

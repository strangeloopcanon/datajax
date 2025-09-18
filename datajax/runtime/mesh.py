"""Utilities for mesh-aware repartitioning and diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
else:
    from collections import abc as _abc

    Sequence = _abc.Sequence


def mesh_shape_from_resource(resources, world_size: int) -> tuple[int, ...]:
    """Derive a mesh shape (axis sizes) from a Resource descriptor."""

    try:
        axes = tuple(getattr(resources, "mesh_axes", ()) or ())
    except Exception:
        axes = ()

    if not axes or len(axes) == 1:
        return (int(world_size),)

    if len(axes) == 2:
        import math

        root = int(math.sqrt(world_size))
        rows = 1
        for r in range(root, 0, -1):
            if world_size % r == 0:
                rows = r
                break
        cols = world_size // rows
        return (int(rows), int(cols))

    return (int(world_size),) + tuple(1 for _ in range(len(axes) - 1))


def resolve_mesh_axis(axis: str | int | None, resources, default: int = 0) -> int:
    """Resolve an axis hint (name or index) against the provided Resource mesh."""

    axes = tuple(getattr(resources, "mesh_axes", ()) or ())
    if isinstance(axis, int):
        if not axes:
            return max(0, axis)
        if axis < 0 or axis >= len(axes):
            raise ValueError(
                f"Axis index {axis} is out of range for mesh axes {axes!r}"
            )
        return axis
    if isinstance(axis, str):
        if axis not in axes:
            raise ValueError(f"Unknown mesh axis {axis!r}; available axes: {axes!r}")
        return axes.index(axis)
    return max(0, min(default, len(axes) - 1 if axes else default))


def compute_destinations_for_mesh(
    hashed: np.ndarray, mesh_shape: Sequence[int], axis_index: int
) -> np.ndarray:
    """Map hash values to rank destinations respecting mesh shape and axis."""

    dims = tuple(int(s) for s in mesh_shape)
    if len(dims) == 0:
        return np.zeros_like(hashed, dtype=np.int32)
    if len(dims) == 1:
        return (hashed % dims[0]).astype(np.int32)

    rem = hashed.copy()
    coords: list[np.ndarray] = [np.zeros_like(hashed, dtype=np.int64)] * len(dims)
    primary = max(0, min(int(axis_index), len(dims) - 1))
    primary_size = max(dims[primary], 1)
    coords[primary] = (rem % primary_size).astype(np.int64)
    rem = rem // primary_size
    for ax, size in enumerate(dims):
        if ax == primary:
            continue
        s = max(size, 1)
        coords[ax] = (rem % s).astype(np.int64)
        rem = rem // s

    dests = np.zeros_like(coords[0], dtype=np.int64)
    mul = 1
    for ax in range(len(dims) - 1, -1, -1):
        dests += coords[ax] * mul
        mul *= max(dims[ax], 1)
    return dests.astype(np.int32)


def compute_destinations_by_key(
    df,
    key: str,
    mesh_shape: Sequence[int],
    axis_index: int,
) -> np.ndarray:
    import pandas as pd

    hashed = pd.util.hash_pandas_object(df[key], index=False).to_numpy(dtype=np.uint64)
    return compute_destinations_for_mesh(hashed, mesh_shape, axis_index)


def rebalance_by_key(df, key: str, mesh_shape: Sequence[int], axis_index: int = 0):
    """Rebalance a pandas DataFrame by hashed key for the given mesh layout."""

    from bodo.libs import distributed_api

    total = 1
    for s in mesh_shape:
        total *= max(int(s), 1)
    if total <= 1:
        return distributed_api.rebalance(df)

    dests = compute_destinations_by_key(df, key, mesh_shape, axis_index)
    return distributed_api.rebalance(df, dests=dests)

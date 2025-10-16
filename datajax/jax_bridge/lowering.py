"""JAX lowering utilities for ExecutionPlan objects.

This module keeps the translation deliberately small: it handles map/project/
filter traces, records sharding metadata, and emits a JAX callable that can be
composed with TPU runtimes such as tpu-inference. Unsupported steps raise
`NotImplementedError` to make the expansion surface explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, MutableMapping

import numpy as np

from datajax.ir.graph import (
    AggregateStep,
    BinaryExpr,
    ColumnRef,
    ComparisonExpr,
    Expr,
    FilterStep,
    InputStep,
    JoinStep,
    Literal,
    LogicalExpr,
    MapStep,
    ProjectStep,
    RenameExpr,
    RepartitionStep,
)
from datajax.runtime.mesh import mesh_shape_from_resource

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import pandas as pd

    from datajax.api.sharding import Resource, ShardSpec
    from datajax.planner.plan import ExecutionPlan
else:
    Sequence = Any
    Resource = ShardSpec = ExecutionPlan = Any
    pd = Any

try:  # pragma: no cover - optional dependency
    import jax
    import jax.numpy as jnp
    from jax import ops
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
except ImportError:  # pragma: no cover - surfaced at runtime
    jax = None
    jnp = None
    ops = None
    Mesh = NamedSharding = PartitionSpec = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LoweredPlan:
    """Result of translating a DataJAX ExecutionPlan to JAX."""

    plan: ExecutionPlan
    mesh: Mesh | None
    in_spec: PartitionSpec | None
    out_spec: PartitionSpec | None
    columns: tuple[str, ...]

    # The callable is stored eagerly to let consumers compose with jit/pjit.
    callable: Any

    def jit(self, **kwargs: Any) -> Any:
        """Return a `jax.jit`-compiled callable for the lowered plan."""

        if jax is None:
            raise RuntimeError("JAX is not installed; cannot jit compile the plan")
        return jax.jit(self.callable, **kwargs)


def _require_jax() -> None:
    if jax is None or jnp is None:  # pragma: no cover - dependent on env
        raise RuntimeError(
            "JAX is required for TPU lowering. Install the optional "
            "extras (e.g. `pip install .[tpu]`) or add JAX to your environment."
        )


def _broadcast_literal(value: Any, reference: Mapping[str, "jnp.ndarray"]) -> "jnp.ndarray":
    if not reference:
        return jnp.asarray(value)
    first = next(iter(reference.values()))
    return jnp.broadcast_to(jnp.asarray(value), first.shape)


def _eval_expr(expr: Expr, columns: Mapping[str, "jnp.ndarray"]) -> "jnp.ndarray":
    if isinstance(expr, ColumnRef):
        if expr.name not in columns:
            raise KeyError(f"Column {expr.name!r} not found in lowering context")
        return columns[expr.name]
    if isinstance(expr, Literal):
        return _broadcast_literal(expr.value, columns)
    if isinstance(expr, RenameExpr):
        return _eval_expr(expr.expr, columns)
    if isinstance(expr, BinaryExpr):
        left = _eval_expr(expr.left, columns)
        right = _eval_expr(expr.right, columns)
        if expr.op == "add":
            return left + right
        if expr.op == "sub":
            return left - right
        if expr.op == "mul":
            return left * right
        if expr.op == "truediv":
            return left / right
        raise NotImplementedError(f"Binary op {expr.op!r} is not supported in JAX lowering")
    if isinstance(expr, ComparisonExpr):
        left = _eval_expr(expr.left, columns)
        right = _eval_expr(expr.right, columns)
        ops = {
            "eq": jnp.equal,
            "ne": jnp.not_equal,
            "lt": jnp.less,
            "le": jnp.less_equal,
            "gt": jnp.greater,
            "ge": jnp.greater_equal,
        }
        fn = ops.get(expr.op)
        if fn is None:
            raise NotImplementedError(f"Comparison op {expr.op!r} not supported")
        return fn(left, right)
    if isinstance(expr, LogicalExpr):
        left = _eval_expr(expr.left, columns)
        right = _eval_expr(expr.right, columns)
        if expr.op == "and":
            return jnp.logical_and(left, right)
        if expr.op == "or":
            return jnp.logical_or(left, right)
        raise NotImplementedError(f"Logical op {expr.op!r} not supported")
    raise NotImplementedError(f"Expression type {type(expr)!r} not implemented")


def _rows_mask(predicate: Expr, columns: Mapping[str, "jnp.ndarray"]) -> "jnp.ndarray":
    mask = _eval_expr(predicate, columns)
    if mask.dtype != jnp.bool_:
        raise TypeError("Filter predicates must evaluate to boolean arrays")
    return mask


def _apply_steps(
    plan: ExecutionPlan,
    columns: MutableMapping[str, "jnp.ndarray"],
) -> tuple[MutableMapping[str, "jnp.ndarray"], tuple[str, ...], Any]:
    target_sharding = None
    for stage in plan.stages:
        for step in stage.steps:
            if isinstance(step, MapStep):
                columns[step.output] = _eval_expr(step.expr, columns)
            elif isinstance(step, ProjectStep):
                columns_keys = {k for k in columns.keys()}
                for name in columns_keys:
                    if name not in step.columns:
                        columns.pop(name)
            elif isinstance(step, FilterStep):
                mask = _rows_mask(step.predicate, columns)
                for key in list(columns.keys()):
                    columns[key] = columns[key][mask]
            elif isinstance(step, RepartitionStep):
                # Data rearrangement happens in runtime; we just track target spec.
                target_sharding = step.spec
            elif isinstance(step, AggregateStep):
                source_columns = dict(columns)
                key_arr = _eval_expr(step.key, source_columns)
                value_arr = _eval_expr(step.value, source_columns)
                if key_arr.ndim != 1 or value_arr.ndim != 1:
                    raise ValueError("Aggregate lowering expects 1D key and value arrays")
                if ops is None:
                    raise RuntimeError("JAX is required to lower aggregate steps")
                unique_keys, inverse, counts = jnp.unique(
                    key_arr, size=None, return_inverse=True, return_counts=True
                )
                num_segments = unique_keys.shape[0]
                if step.agg == "sum":
                    reduced = ops.segment_sum(value_arr, inverse, num_segments)
                elif step.agg == "count":
                    ones = jnp.ones_like(value_arr, dtype=jnp.int32)
                    reduced = ops.segment_sum(ones, inverse, num_segments).astype(jnp.int64)
                elif step.agg == "mean":
                    if np.issubdtype(value_arr.dtype, np.floating):
                        sums = ops.segment_sum(value_arr, inverse, num_segments)
                        denom = jnp.maximum(counts.astype(value_arr.dtype), value_arr.dtype.type(1))
                        reduced = sums / denom
                    else:
                        sums = ops.segment_sum(
                            value_arr.astype(jnp.float32), inverse, num_segments
                        )
                        denom = jnp.maximum(counts.astype(jnp.float32), jnp.float32(1))
                        reduced = sums / denom
                elif step.agg == "min":
                    reduced = ops.segment_min(value_arr, inverse, num_segments)
                elif step.agg == "max":
                    reduced = ops.segment_max(value_arr, inverse, num_segments)
                else:
                    raise NotImplementedError(f"Aggregate op {step.agg!r} is not supported")
                columns.clear()
                columns[step.key_alias] = unique_keys
                columns[step.value_alias] = reduced
            elif isinstance(step, JoinStep):
                if jax is None or jnp is None:
                    raise RuntimeError("JAX is required to lower join steps")
                if step.how != "inner":
                    raise NotImplementedError(f"Join how={step.how!r} is not supported")
                if step.left_on not in columns:
                    raise KeyError(f"Left column {step.left_on!r} missing for join")
                left_key = columns[step.left_on]
                right_df = step.right_data
                right_key = jnp.asarray(right_df[step.right_on].to_numpy())
                right_arrays = {
                    col: jnp.asarray(right_df[col].to_numpy()) for col in step.right_columns
                }
                order = jnp.argsort(right_key)
                sorted_keys = right_key[order]

                def _lookup(value: jnp.ndarray) -> jnp.ndarray:
                    pos = jnp.searchsorted(sorted_keys, value, side="left")
                    in_bounds = (pos < sorted_keys.shape[0]) & (sorted_keys[pos] == value)
                    return jnp.where(in_bounds, order[pos], -1)

                lookup_fn = jax.vmap(_lookup)
                matches = lookup_fn(left_key)
                mask = matches >= 0
                if not jnp.any(mask):
                    for name in list(columns.keys()):
                        columns[name] = columns[name][:0]
                    for col, arr in right_arrays.items():
                        if col == step.left_on and col in columns:
                            continue
                        columns[col] = arr[:0]
                    continue
                for name in list(columns.keys()):
                    columns[name] = columns[name][mask]
                valid_indices = matches[mask]
                for col in step.right_columns:
                    if col == step.left_on and col in columns:
                        continue
                    right_array = right_arrays[col]
                    columns[col] = right_array[valid_indices]
            elif isinstance(step, InputStep):
                # already materialized
                continue
            else:  # pragma: no cover - defensive branch
                raise NotImplementedError(f"Unsupported step {type(step).__name__}")
    return columns, tuple(columns.keys()), target_sharding


def _materialize_mesh(
    resources: Resource | None,
    mesh_devices: Sequence[Any] | None = None,
) -> Mesh | None:
    if resources is None:
        return None
    _require_jax()
    devices = mesh_devices or jax.devices()
    world_size = int(getattr(resources, "world_size", 0) or len(devices))
    if world_size > len(devices):
        raise ValueError(
            f"Resource world_size={world_size} exceeds available devices ({len(devices)})"
        )
    mesh_axes = tuple(getattr(resources, "mesh_axes", ()) or ())
    shape = mesh_shape_from_resource(resources, world_size)
    count = int(np.prod(shape, dtype=int))
    selected = devices[:count]
    if len(selected) != count:
        raise ValueError("Insufficient devices to form requested mesh")
    if len(shape) == 1:
        mesh_array = np.array(selected).reshape(shape[0])
    else:
        mesh_array = np.array(selected).reshape(shape)
    return Mesh(mesh_array, mesh_axes if mesh_axes else ("rows",))


def _partition_from_shard(
    shard_spec: ShardSpec | None,
    mesh: Mesh | None,
) -> PartitionSpec | None:
    if shard_spec is None or mesh is None:
        return None
    axis = getattr(shard_spec, "axis", None)
    if shard_spec.kind == "replicated":
        return PartitionSpec()
    if shard_spec.kind == "key":
        if axis is None:
            # Default to first axis for key partitioning
            axis = mesh.axis_names[0]
        if isinstance(axis, str):
            return PartitionSpec(axis)
        if isinstance(axis, int):
            if axis < 0 or axis >= len(mesh.axis_names):
                raise ValueError("Shard axis index out of range for mesh")
            return PartitionSpec(mesh.axis_names[axis])
    return None


def dataframe_to_device_arrays(df: "pd.DataFrame") -> dict[str, "jnp.ndarray"]:
    """Convert a pandas DataFrame into column-major JAX arrays."""

    _require_jax()
    arrays: dict[str, jnp.ndarray] = {}
    for column in df.columns:
        values = df[column].to_numpy()
        arrays[column] = jnp.asarray(values)
    return arrays


def lower_plan_to_jax(
    plan: ExecutionPlan,
    *,
    sample_df: "pd.DataFrame | None" = None,
    resources: Resource | None = None,
    shard_spec: ShardSpec | None = None,
    mesh_devices: Sequence[Any] | None = None,
) -> LoweredPlan:
    """Translate an ExecutionPlan into a JAX callable and mesh metadata."""

    _require_jax()
    if sample_df is None:
        raise ValueError("sample_df is required to infer shapes for JAX lowering")

    input_arrays = dataframe_to_device_arrays(sample_df)

    def _plan_callable(payload: Mapping[str, Any]) -> dict[str, Any]:
        # Convert payload to mutable columns to allow updates.
        materialized = {name: jnp.asarray(array) for name, array in payload.items()}
        _, keys, _ = _apply_steps(plan, materialized)
        return {key: materialized[key] for key in keys}

    result_columns, column_order, sharding_spec = _apply_steps(plan, dict(input_arrays))

    mesh = _materialize_mesh(resources, mesh_devices)
    out_spec = _partition_from_shard(shard_spec or sharding_spec, mesh)
    in_spec = _partition_from_shard(shard_spec, mesh)

    return LoweredPlan(
        plan=plan,
        mesh=mesh,
        in_spec=in_spec,
        out_spec=out_spec,
        columns=column_order,
        callable=_plan_callable,
    )


__all__ = ["LoweredPlan", "dataframe_to_device_arrays", "lower_plan_to_jax"]

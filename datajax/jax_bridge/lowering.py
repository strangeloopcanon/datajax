"""JAX lowering utilities for ExecutionPlan objects.

This module keeps the translation deliberately small: it handles map/project/
filter traces, records sharding metadata, and emits a JAX callable that can be
composed with TPU runtimes such as tpu-inference. Unsupported steps raise
`NotImplementedError` to make the expansion surface explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING, Any, TypeAlias

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
from datajax.ir.join_semantics import build_join_column_plan
from datajax.runtime.mesh import mesh_shape_from_resource

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, MutableMapping, Sequence

    import pandas as pd

    from datajax.api.sharding import Resource, ShardSpec
    from datajax.planner.plan import ExecutionPlan
else:
    Mapping = MutableMapping = Sequence = Any
    Resource = ShardSpec = ExecutionPlan = Any
    pd = Any

JaxArray: TypeAlias = Any
JaxMesh: TypeAlias = Any
JaxPartitionSpec: TypeAlias = Any


@lru_cache(maxsize=1)
def _require_jax() -> tuple[Any, Any, Any, Any, Any]:
    """Import JAX modules at runtime or raise a helpful error."""

    try:
        jax = import_module("jax")
        jnp = import_module("jax.numpy")
    except Exception as exc:  # pragma: no cover - dependent on env
        raise RuntimeError(
            "JAX is required for TPU lowering. Install the optional extras "
            "(e.g. `pip install .[tpu]`) or add JAX to your environment."
        ) from exc

    jax_ops = getattr(jax, "ops", None)
    if jax_ops is None:  # pragma: no cover - version dependent
        raise RuntimeError("jax.ops is unavailable; install a compatible JAX version")

    sharding = import_module("jax.sharding")
    Mesh = getattr(sharding, "Mesh", None)
    PartitionSpec = getattr(sharding, "PartitionSpec", None)
    if Mesh is None or PartitionSpec is None:  # pragma: no cover - version dependent
        raise RuntimeError(
            "jax.sharding Mesh/PartitionSpec are unavailable; install a compatible "
            "JAX version"
        )

    return jax, jnp, jax_ops, Mesh, PartitionSpec


@dataclass(frozen=True)
class LoweredPlan:
    """Result of translating a DataJAX ExecutionPlan to JAX."""

    plan: ExecutionPlan
    mesh: JaxMesh | None
    in_spec: JaxPartitionSpec | None
    out_spec: JaxPartitionSpec | None
    columns: tuple[str, ...]

    # The callable is stored eagerly to let consumers compose with jit/pjit.
    callable: Any

    def jit(self, **kwargs: Any) -> Any:
        """Return a `jax.jit`-compiled callable for the lowered plan."""
        jax, _, _, _, _ = _require_jax()
        return jax.jit(self.callable, **kwargs)


def _broadcast_literal(
    value: Any,
    reference: Mapping[str, JaxArray],
    *,
    jnp: Any,
) -> JaxArray:
    if not reference:
        return jnp.asarray(value)
    first = next(iter(reference.values()))
    return jnp.broadcast_to(jnp.asarray(value), first.shape)


def _eval_expr(expr: Expr, columns: Mapping[str, JaxArray], *, jnp: Any) -> JaxArray:
    if isinstance(expr, ColumnRef):
        if expr.name not in columns:
            raise KeyError(f"Column {expr.name!r} not found in lowering context")
        return columns[expr.name]
    if isinstance(expr, Literal):
        return _broadcast_literal(expr.value, columns, jnp=jnp)
    if isinstance(expr, RenameExpr):
        return _eval_expr(expr.expr, columns, jnp=jnp)
    if isinstance(expr, BinaryExpr):
        left = _eval_expr(expr.left, columns, jnp=jnp)
        right = _eval_expr(expr.right, columns, jnp=jnp)
        if expr.op == "add":
            return left + right
        if expr.op == "sub":
            return left - right
        if expr.op == "mul":
            return left * right
        if expr.op == "truediv":
            return left / right
        raise NotImplementedError(
            f"Binary op {expr.op!r} is not supported in JAX lowering"
        )
    if isinstance(expr, ComparisonExpr):
        left = _eval_expr(expr.left, columns, jnp=jnp)
        right = _eval_expr(expr.right, columns, jnp=jnp)
        cmp_ops = {
            "eq": jnp.equal,
            "ne": jnp.not_equal,
            "lt": jnp.less,
            "le": jnp.less_equal,
            "gt": jnp.greater,
            "ge": jnp.greater_equal,
        }
        fn = cmp_ops.get(expr.op)
        if fn is None:
            raise NotImplementedError(f"Comparison op {expr.op!r} not supported")
        return fn(left, right)
    if isinstance(expr, LogicalExpr):
        left = _eval_expr(expr.left, columns, jnp=jnp)
        right = _eval_expr(expr.right, columns, jnp=jnp)
        if expr.op == "and":
            return jnp.logical_and(left, right)
        if expr.op == "or":
            return jnp.logical_or(left, right)
        raise NotImplementedError(f"Logical op {expr.op!r} not supported")
    raise NotImplementedError(f"Expression type {type(expr)!r} not implemented")


def _rows_mask(
    predicate: Expr,
    columns: Mapping[str, JaxArray],
    *,
    jnp: Any,
) -> JaxArray:
    mask = _eval_expr(predicate, columns, jnp=jnp)
    if mask.dtype != jnp.bool_:
        raise TypeError("Filter predicates must evaluate to boolean arrays")
    return mask


def _build_join_indices(
    match_matrix: JaxArray,
    *,
    how: str,
    jnp: Any,
) -> tuple[JaxArray, JaxArray]:
    left_match, right_match = jnp.where(match_matrix)
    if how == "inner":
        return left_match, right_match

    left_unmatched = jnp.where(~jnp.any(match_matrix, axis=1))[0]
    right_unmatched = jnp.where(~jnp.any(match_matrix, axis=0))[0]

    if how == "left":
        missing_right = jnp.full(left_unmatched.shape, -1, dtype=right_match.dtype)
        return (
            jnp.concatenate([left_match, left_unmatched]),
            jnp.concatenate([right_match, missing_right]),
        )
    if how == "right":
        missing_left = jnp.full(right_unmatched.shape, -1, dtype=left_match.dtype)
        return (
            jnp.concatenate([left_match, missing_left]),
            jnp.concatenate([right_match, right_unmatched]),
        )
    if how == "outer":
        missing_right = jnp.full(left_unmatched.shape, -1, dtype=right_match.dtype)
        missing_left = jnp.full(right_unmatched.shape, -1, dtype=left_match.dtype)
        return (
            jnp.concatenate([left_match, left_unmatched, missing_left]),
            jnp.concatenate([right_match, missing_right, right_unmatched]),
        )
    raise NotImplementedError(f"Join how={how!r} is not supported")


def _gather_join_output(
    values: JaxArray,
    indices: JaxArray,
    *,
    jnp: Any,
) -> JaxArray:
    if values.ndim != 1:
        raise ValueError("Join lowering expects one-dimensional columns")
    if indices.size == 0:
        return values[:0]

    source = values
    if source.shape[0] == 0:
        source = jnp.zeros((1,), dtype=values.dtype)
    max_index = max(int(source.shape[0]) - 1, 0)
    clipped = jnp.clip(indices, 0, max_index)
    gathered = source[clipped]

    missing = indices < 0
    if not bool(np.asarray(missing).any()):
        return gathered

    if gathered.dtype == jnp.bool_ or jnp.issubdtype(gathered.dtype, jnp.integer):
        promoted = gathered.astype(jnp.float64)
        return jnp.where(missing, jnp.nan, promoted)
    if jnp.issubdtype(gathered.dtype, jnp.floating):
        return jnp.where(missing, jnp.nan, gathered)
    if jnp.issubdtype(gathered.dtype, jnp.complexfloating):
        nan_complex = jnp.asarray(np.nan + 1j * np.nan, dtype=gathered.dtype)
        return jnp.where(missing, nan_complex, gathered)
    raise NotImplementedError(
        f"Join null propagation for dtype {gathered.dtype!r} is unsupported"
    )


def _apply_steps(
    plan: ExecutionPlan,
    columns: MutableMapping[str, JaxArray],
    *,
    jax: Any,
    jnp: Any,
    jax_ops: Any,
) -> tuple[MutableMapping[str, JaxArray], tuple[str, ...], Any]:
    target_sharding = None
    for stage in plan.stages:
        for step in stage.steps:
            if isinstance(step, MapStep):
                columns[step.output] = _eval_expr(step.expr, columns, jnp=jnp)
            elif isinstance(step, ProjectStep):
                columns_keys = {k for k in columns.keys()}
                for name in columns_keys:
                    if name not in step.columns:
                        columns.pop(name)
            elif isinstance(step, FilterStep):
                mask = _rows_mask(step.predicate, columns, jnp=jnp)
                for key in list(columns.keys()):
                    columns[key] = columns[key][mask]
            elif isinstance(step, RepartitionStep):
                # Data rearrangement happens in runtime; we just track target spec.
                target_sharding = step.spec
            elif isinstance(step, AggregateStep):
                source_columns = dict(columns)
                key_arr = _eval_expr(step.key, source_columns, jnp=jnp)
                value_arr = _eval_expr(step.value, source_columns, jnp=jnp)
                if key_arr.ndim != 1 or value_arr.ndim != 1:
                    raise ValueError(
                        "Aggregate lowering expects 1D key and value arrays"
                    )
                unique_keys, inverse, counts = jnp.unique(
                    key_arr, size=None, return_inverse=True, return_counts=True
                )
                num_segments = unique_keys.shape[0]
                if step.agg == "sum":
                    reduced = jax_ops.segment_sum(value_arr, inverse, num_segments)
                elif step.agg == "count":
                    ones = jnp.ones_like(value_arr, dtype=jnp.int32)
                    reduced = jax_ops.segment_sum(ones, inverse, num_segments).astype(
                        jnp.int64
                    )
                elif step.agg == "mean":
                    if np.issubdtype(value_arr.dtype, np.floating):
                        sums = jax_ops.segment_sum(value_arr, inverse, num_segments)
                        denom = jnp.maximum(
                            counts.astype(value_arr.dtype),
                            value_arr.dtype.type(1),
                        )
                        reduced = sums / denom
                    else:
                        sums = jax_ops.segment_sum(
                            value_arr.astype(jnp.float32), inverse, num_segments
                        )
                        denom = jnp.maximum(counts.astype(jnp.float32), jnp.float32(1))
                        reduced = sums / denom
                elif step.agg == "min":
                    reduced = jax_ops.segment_min(value_arr, inverse, num_segments)
                elif step.agg == "max":
                    reduced = jax_ops.segment_max(value_arr, inverse, num_segments)
                else:
                    raise NotImplementedError(
                        f"Aggregate op {step.agg!r} is not supported"
                    )
                columns.clear()
                columns[step.key_alias] = unique_keys
                columns[step.value_alias] = reduced
            elif isinstance(step, JoinStep):
                missing_left_keys = [key for key in step.left_on if key not in columns]
                if missing_left_keys:
                    raise KeyError(
                        f"Left columns {missing_left_keys!r} missing for join"
                    )
                right_df = step.right_data
                if right_df is None:
                    raise ValueError("Join step is missing right-hand side data")
                missing_right_keys = [
                    key for key in step.right_on if key not in right_df.columns
                ]
                if missing_right_keys:
                    raise KeyError(
                        f"Right columns {missing_right_keys!r} missing for join"
                    )
                source_columns = dict(columns)
                left_key_arrays = [source_columns[key] for key in step.left_on]
                right_key_arrays = [
                    jnp.asarray(right_df[key].to_numpy()) for key in step.right_on
                ]
                if not left_key_arrays:
                    raise ValueError("Join step has no key columns")
                match_matrix = jnp.ones(
                    (left_key_arrays[0].shape[0], right_key_arrays[0].shape[0]),
                    dtype=jnp.bool_,
                )
                for left_key, right_key in zip(
                    left_key_arrays, right_key_arrays, strict=True
                ):
                    match_matrix = jnp.logical_and(
                        match_matrix,
                        left_key[:, None] == right_key[None, :],
                    )
                left_indices, right_indices = _build_join_indices(
                    match_matrix,
                    how=step.how,
                    jnp=jnp,
                )

                right_arrays = {
                    col: jnp.asarray(right_df[col].to_numpy())
                    for col in step.right_columns
                }
                column_plan = build_join_column_plan(
                    left_columns=tuple(source_columns.keys()),
                    right_columns=step.right_columns,
                    left_on=step.left_on,
                    right_on=step.right_on,
                    suffixes=step.suffixes,
                )
                joined_columns: dict[str, JaxArray] = {}
                for source_name, out_name in column_plan.left_pairs:
                    joined_columns[out_name] = _gather_join_output(
                        source_columns[source_name],
                        left_indices,
                        jnp=jnp,
                    )
                for _, source_name, out_name in column_plan.right_pairs:
                    joined_columns[out_name] = _gather_join_output(
                        right_arrays[source_name],
                        right_indices,
                        jnp=jnp,
                    )
                columns.clear()
                columns.update(joined_columns)
            elif isinstance(step, InputStep):
                # already materialized
                continue
            else:  # pragma: no cover - defensive branch
                raise NotImplementedError(f"Unsupported step {type(step).__name__}")
    return columns, tuple(columns.keys()), target_sharding


def _materialize_mesh(
    resources: Resource | None,
    mesh_devices: Sequence[Any] | None = None,
) -> JaxMesh | None:
    if resources is None:
        return None
    jax, _, _, Mesh, _ = _require_jax()
    devices = mesh_devices or jax.devices()
    world_size = int(getattr(resources, "world_size", 0) or len(devices))
    if world_size > len(devices):
        raise ValueError(
            f"Resource world_size={world_size} exceeds available devices "
            f"({len(devices)})"
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
    mesh: JaxMesh | None,
) -> JaxPartitionSpec | None:
    if shard_spec is None or mesh is None:
        return None
    _, _, _, _, PartitionSpec = _require_jax()
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


def dataframe_to_device_arrays(df: pd.DataFrame) -> dict[str, JaxArray]:
    """Convert a pandas DataFrame into column-major JAX arrays."""

    _, jnp, _, _, _ = _require_jax()
    arrays: dict[str, JaxArray] = {}
    for column in df.columns:
        values = df[column].to_numpy()
        arrays[column] = jnp.asarray(values)
    return arrays


def lower_plan_to_jax(
    plan: ExecutionPlan,
    *,
    sample_df: pd.DataFrame | None = None,
    resources: Resource | None = None,
    shard_spec: ShardSpec | None = None,
    mesh_devices: Sequence[Any] | None = None,
) -> LoweredPlan:
    """Translate an ExecutionPlan into a JAX callable and mesh metadata."""

    jax, jnp, jax_ops, _, _ = _require_jax()
    if sample_df is None:
        raise ValueError("sample_df is required to infer shapes for JAX lowering")

    input_arrays = dataframe_to_device_arrays(sample_df)

    def _plan_callable(payload: Mapping[str, Any]) -> dict[str, Any]:
        # Convert payload to mutable columns to allow updates.
        materialized = {name: jnp.asarray(array) for name, array in payload.items()}
        _, keys, _ = _apply_steps(plan, materialized, jax=jax, jnp=jnp, jax_ops=jax_ops)
        return {key: materialized[key] for key in keys}

    _, column_order, sharding_spec = _apply_steps(
        plan, dict(input_arrays), jax=jax, jnp=jnp, jax_ops=jax_ops
    )

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

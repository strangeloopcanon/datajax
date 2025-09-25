"""Plan metrics and heuristics for offline tuning.

These estimates are intentionally coarse. They provide offline hooks for
external tuners (e.g., BCache/hotweights) to reason about bytes moved,
column reuse, and to derive suggested packing/tile orders.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from datajax.ir.graph import (
    AggregateStep,
    BinaryExpr,
    ColumnRef,
    ComparisonExpr,
    Expr,
    FilterStep,
    InputStep,
    JoinStep,
    LogicalExpr,
    MapStep,
    ProjectStep,
    RenameExpr,
    RepartitionStep,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    import pandas as pd

    from datajax.planner.plan import ExecutionPlan
else:  # pragma: no cover - runtime types
    import types as _types
    from collections import abc as _abc

    Mapping = _abc.Mapping
    Sequence = _abc.Sequence
    pd = _types.SimpleNamespace(DataFrame=object)


def _expr_columns(expr: Expr) -> set[str]:
    if isinstance(expr, ColumnRef):
        return {expr.name}
    if isinstance(expr, RenameExpr):
        return _expr_columns(expr.expr)
    if isinstance(expr, BinaryExpr):
        return _expr_columns(expr.left) | _expr_columns(expr.right)
    if isinstance(expr, ComparisonExpr):
        return _expr_columns(expr.left) | _expr_columns(expr.right)
    if isinstance(expr, LogicalExpr):
        return _expr_columns(expr.left) | _expr_columns(expr.right)
    return set()


@dataclass
class PlanMetrics:
    # Structure
    steps_total: int
    transform_steps: int
    join_steps: int
    aggregate_steps: int
    repartition_steps: int

    # Scale estimates
    estimated_input_rows: int | None = None
    estimated_row_size_bytes: int | None = None
    estimated_input_bytes: int | None = None
    estimated_shuffle_bytes: int | None = None

    # GPU-centric knobs (heuristics only)
    expected_wgmma_occupancy: float | None = None
    predicted_l2_reuse: float | None = None

    # Hints for external tuners
    pack_order_hint: tuple[str, ...] = ()
    column_usage_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # Runtime overrides (populated when counters from production are available)
    runtime_input_bytes: int | None = None
    runtime_shuffle_bytes: int | None = None
    runtime_wgmma_occupancy: float | None = None
    runtime_l2_reuse: float | None = None
    runtime_notes: list[str] = field(default_factory=list)


def _approx_row_size_bytes(df: pd.DataFrame | None, columns: Sequence[str]) -> int:
    if df is None:
        # Fallback: assume 16 bytes per column (rough heuristic for numeric)
        return 16 * len(columns)
    total = 0
    for c in columns:
        try:
            dtype = df.dtypes[c]
        except Exception:
            total += 16
            continue
        try:
            itemsize = getattr(dtype, "itemsize", None)
            if itemsize is None:
                # object/Arrow-backed: assume 16 bytes per value
                total += 16
            else:
                total += int(itemsize)
        except Exception:
            total += 16
    return total


def _collect_column_usage(trace: Sequence[object]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for step in trace:
        cols: set[str] = set()
        if isinstance(step, MapStep):
            cols = _expr_columns(step.expr)
            # Include output column as well (it will be referenced later)
            cols.add(step.output)
        elif isinstance(step, FilterStep):
            cols = _expr_columns(step.predicate)
        elif isinstance(step, ProjectStep):
            cols = set(step.columns)
        elif isinstance(step, AggregateStep):
            cols = _expr_columns(step.key) | _expr_columns(step.value)
            cols.add(step.key_alias)
            cols.add(step.value_alias)
        elif isinstance(step, JoinStep):
            cols.add(step.left_on)
            cols.add(step.right_on)
            cols |= set(step.right_columns)
        for c in cols:
            usage[c] = usage.get(c, 0) + 1
    return usage


def estimate_plan_metrics(
    plan: ExecutionPlan,
    *,
    sample_df: pd.DataFrame | None = None,
) -> PlanMetrics:
    """Estimate coarse metrics for a compiled plan.

    Parameters
    ----------
    plan:
        The ExecutionPlan to analyze.
    sample_df:
        Optional concrete input frame used for scale estimates.
    """

    # Structure counts
    stages: Sequence[Any] = tuple(getattr(plan, "stages", ()) or ())
    trace: Sequence[object] = tuple(getattr(plan, "trace", ()) or ())
    resources = getattr(plan, "resources", None)

    transform_steps = 0
    join_steps = 0
    aggregate_steps = 0
    repartition_steps = 0

    if stages:
        for stage in stages:
            for step in getattr(stage, "steps", ()):  # type: ignore[attr-defined]
                if isinstance(step, (MapStep, FilterStep, ProjectStep)):
                    transform_steps += 1
                elif isinstance(step, JoinStep):
                    join_steps += 1
                elif isinstance(step, AggregateStep):
                    aggregate_steps += 1
                elif isinstance(step, RepartitionStep):
                    repartition_steps += 1
    else:
        for step in trace:
            if isinstance(step, (MapStep, FilterStep, ProjectStep)):
                transform_steps += 1
            elif isinstance(step, JoinStep):
                join_steps += 1
            elif isinstance(step, AggregateStep):
                aggregate_steps += 1
            elif isinstance(step, RepartitionStep):
                repartition_steps += 1

    if stages:
        steps_total = sum(len(getattr(stage, "steps", ())) for stage in stages)
    else:
        steps_total = len(trace)

    # Row/byte estimates
    estimated_input_rows = None
    if sample_df is not None:
        try:
            estimated_input_rows = int(len(sample_df))
        except Exception:
            estimated_input_rows = None

    input_schema: tuple[str, ...] = ()
    if stages:
        first_stage = stages[0]
        input_schema = tuple(getattr(first_stage, "input_schema", ()) or ())
    elif trace:
        first_step = trace[0]
        if isinstance(first_step, InputStep):
            input_schema = tuple(first_step.schema)
    if not input_schema and sample_df is not None:
        input_schema = tuple(sample_df.columns)
    row_size = _approx_row_size_bytes(sample_df, input_schema)
    estimated_input_bytes = (
        row_size * estimated_input_rows if estimated_input_rows is not None else None
    )

    # Approximate shuffle bytes from repartitions and non-colocated joins
    world_size = getattr(resources, "world_size", 1) or 1
    shuffle_events = repartition_steps + join_steps
    estimated_shuffle_bytes = None
    if estimated_input_rows is not None:
        # Assume a full-row shuffle per event; scale by (world_size - 1)/world_size
        factor = max(world_size - 1, 0) / max(world_size, 1)
        estimated_shuffle_bytes = int(
            row_size * estimated_input_rows * shuffle_events * factor
        )

    # Column usage for pack order and reuse score
    usage = _collect_column_usage(trace)
    pack_order_hint = tuple(
        k for k, _ in sorted(usage.items(), key=lambda kv: (-kv[1], kv[0]))
    )

    # Heuristic GPU-centric numbers (placeholders for downstream tuners)
    # More arithmetic (MapSteps) over less shuffle tends to increase occupancy
    if estimated_shuffle_bytes is not None:
        denom = max(estimated_shuffle_bytes, 1)
        occ = min(1.0, (transform_steps + 1) / (1.0 + denom / (row_size + 1)))
    else:
        occ = min(1.0, (transform_steps + 1) / 8.0)

    # L2 reuse proxy: normalized concentration of column references
    if usage:
        total_refs = sum(usage.values())
        top_refs = max(usage.values())
        l2_reuse = min(1.0, top_refs / max(total_refs, 1))
    else:
        l2_reuse = None

    notes: list[str] = []
    if repartition_steps:
        notes.append(f"repartitions={repartition_steps}")
    if join_steps:
        notes.append(f"joins={join_steps}")
    if aggregate_steps:
        notes.append(f"aggregations={aggregate_steps}")

    return PlanMetrics(
        steps_total=steps_total,
        transform_steps=transform_steps,
        join_steps=join_steps,
        aggregate_steps=aggregate_steps,
        repartition_steps=repartition_steps,
        estimated_input_rows=estimated_input_rows,
        estimated_row_size_bytes=row_size,
        estimated_input_bytes=estimated_input_bytes,
        estimated_shuffle_bytes=estimated_shuffle_bytes,
        expected_wgmma_occupancy=occ,
        predicted_l2_reuse=l2_reuse,
        pack_order_hint=pack_order_hint,
        column_usage_counts=usage,
        notes=notes,
    )


def merge_runtime_counters(metrics: PlanMetrics, counters: Mapping[str, Any]) -> None:
    """Overlay runtime counters onto estimated metrics."""

    input_bytes = counters.get("bytes_moved") or counters.get("input_bytes")
    if input_bytes is not None:
        metrics.runtime_input_bytes = int(input_bytes)
    shuffle_bytes = counters.get("shuffle_bytes") or counters.get("bytes_shuffled")
    if shuffle_bytes is not None:
        metrics.runtime_shuffle_bytes = int(shuffle_bytes)
    occ = counters.get("wgmma_occupancy") or counters.get("occupancy")
    if occ is not None:
        metrics.runtime_wgmma_occupancy = float(occ)
    reuse = counters.get("l2_reuse") or counters.get("cache_reuse")
    if reuse is not None:
        metrics.runtime_l2_reuse = float(reuse)
    runtime_notes = counters.get("notes")
    if isinstance(runtime_notes, (list, tuple)):
        metrics.runtime_notes = [str(item) for item in runtime_notes]


def metrics_to_dict(metrics: PlanMetrics | None) -> dict[str, Any] | None:
    """Convert PlanMetrics into a JSON-friendly dictionary."""

    if metrics is None:
        return None
    data = asdict(metrics)
    # Convert non-serializable types if necessary
    if isinstance(data.get("pack_order_hint"), tuple):
        data["pack_order_hint"] = list(data["pack_order_hint"])
    if isinstance(data.get("notes"), tuple):  # defensive
        data["notes"] = list(data["notes"])
    if isinstance(data.get("runtime_notes"), tuple):
        data["runtime_notes"] = list(data["runtime_notes"])
    return data


__all__ = [
    "PlanMetrics",
    "estimate_plan_metrics",
    "merge_runtime_counters",
    "metrics_to_dict",
]

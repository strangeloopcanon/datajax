"""Trace replay and policy suggestion utilities for offline tuning.

This module lets you:
  - serialize/deserialize traces (via datajax.ir.serialize),
  - rebuild a plan from a stored trace and an input sample,
  - compute coarse metrics and derive staged policy suggestions
    (BM/BN/BK tile sizes, swizzle size, stage depth) for external tuners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from datajax.ir.serialize import trace_from_list
from datajax.planner.metrics import PlanMetrics, estimate_plan_metrics
from datajax.planner.plan import build_plan

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping, Sequence

    import pandas as pd
else:  # pragma: no cover
    import types as _types
    from collections import abc as _abc

    Iterable = _abc.Iterable
    Mapping = _abc.Mapping
    Sequence = _abc.Sequence
    pd = _types.SimpleNamespace(DataFrame=object)


@dataclass(frozen=True)
class StagedPolicy:
    BM: int
    BN: int
    BK: int
    swizzle_size: int
    stage_depth: int
    notes: tuple[str, ...] = ()


def suggest_policies_from_metrics(metrics: PlanMetrics) -> StagedPolicy:
    """Derive simple policy knobs from plan metrics.

    Heuristics (placeholder):
      - Higher estimated reuse → larger swizzle/tile sizes.
      - Higher occupancy proxy → prefer larger BM/BN; BK scales modestly.
      - Stage depth roughly bounded by number of transform steps.
    """

    occ = metrics.expected_wgmma_occupancy or 0.5
    reuse = metrics.predicted_l2_reuse or 0.25

    # Tile sizes: ramp up with occupancy/reuse
    def _bucket(v: float, lo: int, hi: int) -> int:
        v = max(0.0, min(1.0, v))
        return int(lo + (hi - lo) * v)

    BM = _bucket(occ, 64, 256)
    BN = _bucket(occ, 64, 256)
    BK = _bucket((occ + reuse) / 2.0, 16, 128)
    swizzle = 32 if reuse < 0.5 else 64
    depth = max(1, min(metrics.transform_steps // 2 + 1, 6))

    notes = list(metrics.notes)
    if metrics.estimated_shuffle_bytes:
        notes.append(f"shuffle_bytes≈{metrics.estimated_shuffle_bytes}")
    return StagedPolicy(
        BM=BM,
        BN=BN,
        BK=BK,
        swizzle_size=swizzle,
        stage_depth=depth,
        notes=tuple(notes),
    )


def replay_and_tune(
    trace_or_serialized: Sequence[object] | Iterable[dict[str, Any]],
    *,
    input_df: pd.DataFrame,
    resources: Any | None = None,
    backend: str = "pandas",
    mode: str = "stub",
    rhs_tables: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[PlanMetrics, StagedPolicy]:
    """Rebuild plan and derive metrics/policy suggestions."""

    # Detect serialized form (sequence of dicts)
    items = list(trace_or_serialized)
    if items and isinstance(items[0], dict):
        trace = trace_from_list(items, rhs_tables=rhs_tables)
    else:
        trace = items  # type: ignore[assignment]

    plan = build_plan(
        trace,
        backend=backend,
        mode=mode,
        input_df=input_df,
        resources=resources,
    )
    # If native plan is returned, metrics are not embedded; derive from trace
    try:
        metrics = getattr(plan, "metrics", None)  # type: ignore[attr-defined]
    except Exception:
        metrics = None
    if metrics is None:
        from datajax.planner.plan import ExecutionPlan

        if hasattr(plan, "trace") and hasattr(plan, "stages"):
            metrics = estimate_plan_metrics(plan)  # type: ignore[arg-type]
        else:
            # Fallback: build a lightweight ExecutionPlan-like holder
            dummy = ExecutionPlan(
                backend=backend,
                mode=mode,
                stages=(),
                trace=tuple(trace),
                final_schema=tuple(input_df.columns),
                final_sharding=None,
                resources=resources,
            )
            metrics = estimate_plan_metrics(dummy, sample_df=input_df)

    policy = suggest_policies_from_metrics(metrics)
    return metrics, policy


__all__ = ["StagedPolicy", "suggest_policies_from_metrics", "replay_and_tune"]

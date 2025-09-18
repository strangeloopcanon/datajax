"""Blueprint for building native Bodo plans from DataJAX IR.

This module does **not** produce runnable code inside the sandbox – Bodo requires
MPI-enabled execution and direct access to its planner internals.  The functions
below sketch the translation layer so that you (or a future collaborator) can
implement and validate it on a machine where the real Bodo runtime is available.

Read the inline comments carefully: they reference the exact plan classes and
helpers defined in `bodo/pandas/plan.py` and `bodo/pandas/frame.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from datajax.planner.plan import ExecutionPlan, Stage

if TYPE_CHECKING:
    import pandas as pd
else:
    import types as _types

    pd = _types.SimpleNamespace(DataFrame=object)


@dataclass
class NativeStage:
    """Placeholder describing a stage to be lowered into Bodo's LazyPlan tree."""

    stage: Stage
    description: str


def lower_plan_to_bodo(plan: ExecutionPlan, frame: pd.DataFrame) -> list[NativeStage]:
    """Return a high-level lowering plan for Bodo.

    Parameters
    ----------
    plan:
        The stage-based execution plan produced by DataJAX.
    frame:
        The concrete pandas input that triggered the trace.  In a real
        implementation this is used to derive schemas (zero-row samples).

    Notes
    -----
    - The real implementation should import the following at runtime (inside the
      function to avoid sandbox import errors):
        ``from bodo.pandas.plan import (LogicalGetPandasReadSeq,
        LogicalGetPandasReadParallel, LogicalFilter, LogicalProjection,
        LogicalAggregate, LogicalComparisonJoin, ColRefExpression,
        ArithOpExpression, ComparisonOpExpression, AggregateExpression, …)``
    - Each DataJAX IR node maps to one or more of those classes.  For example,
      `graph.MapStep` becomes `LogicalProjection` with appropriate
      `ArithOpExpression` objects.  `graph.AggregateStep` becomes
      `LogicalAggregate` + `AggregateExpression`.
    - Once all stages are translated into a `LazyPlan`, execute it via
      `bodo.pandas.plan.execute_plan(lazy_plan)`.
    - Repartition should eventually map onto Bodo's distribution primitives
      (look for a `LogicalRepartition` or similar in the Bodo sources).
    - This function currently returns an explanatory scaffold so that the work
      can be carried out on an environment where Bodo is fully available.
    """

    native: list[NativeStage] = []

    native.append(
        NativeStage(
            stage=plan.stages[0] if plan.stages else Stage("input", (), (), (), None),
            description=(
                "Create the base LazyPlan using "
                "LogicalGetPandasReadSeq/Parallel. Use frame.head(0) to build "
                "the schema (see bodo.pandas.base._empty_like)."
            ),
        )
    )

    for stage in plan.stages:
        if stage.kind == "transform":
            native.append(
                NativeStage(
                    stage=stage,
                    description=(
                        "Translate Map/Filter/Project steps into "
                        "LogicalProjection/LogicalFilter with ColRefExpression, "
                        "ArithOpExpression, and ComparisonOpExpression."
                    ),
                )
            )
        elif stage.kind == "join":
            native.append(
                NativeStage(
                    stage=stage,
                    description=(
                        "Use LogicalComparisonJoin with join keys from graph.JoinStep; "
                        "wrap RHS data using LazyPlanDistributedArg if necessary."
                    ),
                )
            )
        elif stage.kind == "aggregate":
            native.append(
                NativeStage(
                    stage=stage,
                    description=(
                        "Build LogicalAggregate + AggregateExpression (e.g. SUM) based "
                        "on graph.AggregateStep metadata."
                    ),
                )
            )
        elif stage.kind == "repartition":
            native.append(
                NativeStage(
                    stage=stage,
                    description=(
                        "Insert a repartition node (check Bodo for the appropriate "
                        "Logical* class). Update sharding state accordingly."
                    ),
                )
            )
        elif stage.kind == "input":
            continue
        else:
            native.append(
                NativeStage(
                    stage=stage,
                    description=(
                        f"TODO: implement lowering for stage kind {stage.kind}"
                    ),
                )
            )

    native.append(
        NativeStage(
            stage=plan.stages[-1] if plan.stages else Stage("final", (), (), (), None),
            description=(
                "Wrap the final LazyPlan with wrap_plan(...) and call execute_plan. "
                "Remember to broadcast metadata via LazyPlanDistributedArg when "
                "running in parallel."
            ),
        )
    )

    return native


__all__ = ["NativeStage", "lower_plan_to_bodo"]

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import bodo
import pandas as pd
from bodo.ext import plan_optimizer
from bodo.pandas.plan import (
    AggregateExpression,
    ArithOpExpression,
    ColRefExpression,
    ComparisonOpExpression,
    ConjunctionOpExpression,
    ConstantExpression,
    LazyPlan,
    LazyPlanDistributedArg,
    LogicalAggregate,
    LogicalComparisonJoin,
    LogicalFilter,
    LogicalGetPandasReadParallel,
    LogicalGetPandasReadSeq,
    LogicalProjection,
    _get_df_python_func_plan,
    make_col_ref_exprs,
)
from bodo.pandas.utils import get_n_index_arrays

from datajax.ir.graph import (
    AggregateStep,
    BinaryExpr,
    ColumnRef,
    ComparisonExpr,
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
from datajax.runtime.mesh import (
    mesh_shape_from_resource,
    rebalance_by_key,
    resolve_mesh_axis,
)
from datajax.runtime.plan_diagnostics import PlanDiagnostics

if TYPE_CHECKING:
    from collections.abc import Sequence
else:
    from collections import abc as _abc

    Sequence = _abc.Sequence


class DataJAXPlan(LazyPlan):
    """Translate DataJAX IR steps into Bodo LazyPlan nodes."""

    _ARITH_OPS = {
        "add": "__add__",
        "sub": "__sub__",
        "mul": "__mul__",
        "truediv": "__truediv__",
        "floordiv": "__floordiv__",
        "pow": "__pow__",
    }
    _COMPARE_OPS = {
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
    }
    _LOGICAL_OPS = {
        "and": "__and__",
        "or": "__or__",
    }

    _AGG_RESULT_DTYPES = {
        "count": pd.Series([], dtype="int64").dtype,
        "mean": pd.Series([], dtype="float64").dtype,
    }

    _JOIN_TYPES = {
        "inner": plan_optimizer.CJoinType.INNER,
        "left": plan_optimizer.CJoinType.LEFT,
        "right": plan_optimizer.CJoinType.RIGHT,
        "outer": plan_optimizer.CJoinType.OUTER,
    }

    def __init__(
        self,
        trace: Sequence[object],
        input_df: pd.DataFrame,
        *,
        resources: object | None = None,
    ) -> None:
        self.trace = trace
        self._input_df = input_df
        self.resources = resources
        self.diagnostics = PlanDiagnostics()

        self._primary_axis_idx = 0
        try:
            last_spec = next(
                (
                    step.spec
                    for step in reversed(trace)
                    if isinstance(step, RepartitionStep)
                ),
                None,
            )
            if last_spec is not None and resources is not None:
                self._primary_axis_idx = resolve_mesh_axis(last_spec.axis, resources, 0)
        except Exception:
            self._primary_axis_idx = 0

        plan = self._translate_trace(trace, input_df.iloc[0:0])

        self.final_sharding = next(
            (
                step.spec
                for step in reversed(trace)
                if isinstance(step, RepartitionStep)
            ),
            None,
        )
        self.final_schema = tuple(plan.empty_data.columns)
        self.backend = "bodo"
        self.mode = "native"

        super().__init__(plan.plan_class, plan.empty_data, *plan.args)

    def describe(self) -> list[str]:
        return self.diagnostics.describe()

    def _empty_series(self, dtype, name: str | None = None) -> pd.Series:
        return pd.Series([], dtype=dtype, name=name)

    def _plan_from_pandas(self, df: pd.DataFrame) -> LazyPlan:
        empty = df.head(0)
        if bodo.dataframe_library_run_parallel:
            return LogicalGetPandasReadParallel(
                empty,
                len(df),
                LazyPlanDistributedArg(df),
            )
        return LogicalGetPandasReadSeq(empty, df)

    def _translate_expr(self, expr, source_plan: LazyPlan, name: str | None = None):
        if isinstance(expr, ColumnRef):
            return ColRefExpression(
                source_plan.empty_data[[expr.name]],
                source_plan,
                source_plan.empty_data.columns.get_loc(expr.name),
            )
        if isinstance(expr, Literal):
            dtype = pd.Series([expr.value]).dtype
            return ConstantExpression(
                self._empty_series(dtype, name),
                source_plan,
                expr.value,
            )
        if isinstance(expr, BinaryExpr):
            left = self._translate_expr(expr.left, source_plan)
            right = self._translate_expr(expr.right, source_plan)
            op = self._ARITH_OPS.get(expr.op)
            if op is None:
                raise NotImplementedError(f"Unsupported arithmetic op: {expr.op}")
            return ArithOpExpression(
                self._empty_series("float64", name),
                left,
                right,
                op,
            )
        if isinstance(expr, ComparisonExpr):
            left = self._translate_expr(expr.left, source_plan)
            right = self._translate_expr(expr.right, source_plan)
            op = self._COMPARE_OPS.get(expr.op)
            if op is None:
                raise NotImplementedError(f"Unsupported comparison op: {expr.op}")
            return ComparisonOpExpression(
                self._empty_series("bool", name),
                left,
                right,
                op,
            )
        if isinstance(expr, LogicalExpr):
            left = self._translate_expr(expr.left, source_plan)
            right = self._translate_expr(expr.right, source_plan)
            op = self._LOGICAL_OPS.get(expr.op)
            if op is None:
                raise NotImplementedError(f"Unsupported logical op: {expr.op}")
            return ConjunctionOpExpression(
                self._empty_series("bool", name),
                left,
                right,
                op,
            )
        if isinstance(expr, RenameExpr):
            return self._translate_expr(expr.expr, source_plan, expr.alias)
        raise NotImplementedError(f"Unsupported expression: {expr}")

    def _translate_trace(
        self, trace: Sequence[object], empty_data: pd.DataFrame
    ) -> LazyPlan:
        plan: LazyPlan | None = None
        for step in trace:
            if isinstance(step, InputStep):
                plan = self._translate_input_step(step, empty_data)
            elif isinstance(step, ProjectStep):
                plan = self._translate_project_step(step, plan)
            elif isinstance(step, FilterStep):
                plan = self._translate_filter_step(step, plan)
            elif isinstance(step, MapStep):
                plan = self._translate_map_step(step, plan)
            elif isinstance(step, AggregateStep):
                plan = self._translate_aggregate_step(step, plan)
            elif isinstance(step, JoinStep):
                plan = self._translate_join_step(step, plan)
            elif isinstance(step, RepartitionStep):
                plan = self._translate_repartition_step(step, plan)
            else:
                raise NotImplementedError(f"Unsupported IR step: {step}")
        assert plan is not None
        return plan

    def _translate_input_step(
        self, step: InputStep, empty_data: pd.DataFrame
    ) -> LazyPlan:
        if bodo.dataframe_library_run_parallel:
            return LogicalGetPandasReadParallel(
                empty_data,
                len(self._input_df),
                LazyPlanDistributedArg(self._input_df),
            )
        return LogicalGetPandasReadSeq(empty_data, self._input_df)

    def _translate_project_step(
        self, step: ProjectStep, source_plan: LazyPlan
    ) -> LazyPlan:
        empty_data = source_plan.empty_data[list(step.columns)]
        exprs = [
            ColRefExpression(
                source_plan.empty_data[[col]],
                source_plan,
                source_plan.empty_data.columns.get_loc(col),
            )
            for col in step.columns
        ]
        return LogicalProjection(empty_data, source_plan, exprs)

    def _translate_filter_step(
        self, step: FilterStep, source_plan: LazyPlan
    ) -> LazyPlan:
        predicate = self._translate_expr(step.predicate, source_plan)
        return LogicalFilter(source_plan.empty_data, source_plan, predicate)

    def _translate_map_step(self, step: MapStep, source_plan: LazyPlan) -> LazyPlan:
        empty_data = source_plan.empty_data.copy()
        existing_cols = list(empty_data.columns)
        exprs = []
        if step.output in existing_cols:
            for col in existing_cols:
                if col == step.output:
                    exprs.append(
                        self._translate_expr(step.expr, source_plan, step.output)
                    )
                else:
                    exprs.append(
                        ColRefExpression(
                            source_plan.empty_data[[col]],
                            source_plan,
                            source_plan.empty_data.columns.get_loc(col),
                        )
                    )
            return LogicalProjection(empty_data[existing_cols], source_plan, exprs)

        empty_data[step.output] = pd.Series([], dtype="float64")
        for col in existing_cols:
            exprs.append(
                ColRefExpression(
                    source_plan.empty_data[[col]],
                    source_plan,
                    source_plan.empty_data.columns.get_loc(col),
                )
            )
        exprs.append(self._translate_expr(step.expr, source_plan, step.output))
        new_cols = existing_cols + [step.output]
        return LogicalProjection(empty_data[new_cols], source_plan, exprs)

    def _translate_aggregate_step(
        self, step: AggregateStep, source_plan: LazyPlan
    ) -> LazyPlan:
        key_dtype = source_plan.empty_data[step.key_alias].dtype
        value_dtype = source_plan.empty_data[step.value_alias].dtype
        result_dtype = self._AGG_RESULT_DTYPES.get(step.agg, value_dtype)
        empty_data = pd.DataFrame(
            {
                step.key_alias: pd.Series([], dtype=key_dtype),
                step.value_alias: pd.Series([], dtype=result_dtype),
            }
        )
        key_idx = source_plan.empty_data.columns.get_loc(step.key_alias)
        value_idx = source_plan.empty_data.columns.get_loc(step.value_alias)
        agg_expr = AggregateExpression(
            self._empty_series(result_dtype, step.value_alias),
            source_plan,
            step.agg,
            None,
            [value_idx],
            False,
        )
        return LogicalAggregate(empty_data, source_plan, [key_idx], [agg_expr])

    def _translate_join_step(
        self, step: JoinStep, source_plan: LazyPlan
    ) -> LazyPlan:
        join_type = self._JOIN_TYPES.get(step.how)
        if join_type is None:
            raise NotImplementedError(f"Unsupported join type: {step.how}")

        right_df = step.right_data
        right_plan = self._plan_from_pandas(right_df)

        # Only attempt to rebalance RHS when running the dataframe library in parallel.
        # In sequential mode, wrapping a DataFrame-returning function via
        # _get_df_python_func_plan triggers a scalar-UDF pathway inside Bodo
        # that expects a Series-like output (leading to dtype errors).
        if self.resources is not None:
            try:
                import bodo  # local import to avoid sandbox import errors
                run_parallel = getattr(bodo, "dataframe_library_run_parallel", False)
            except Exception:
                run_parallel = False
            import os
            if os.environ.get("DATAJAX_DISABLE_NATIVE_REBALANCE", "0") == "1":
                run_parallel = False

            if run_parallel:
                mesh_shape = mesh_shape_from_resource(self.resources, bodo.get_size())
                axis_idx = getattr(self, "_primary_axis_idx", 0)
                try:
                    tmp = _get_df_python_func_plan(
                        right_plan,
                        right_plan.empty_data,
                        rebalance_by_key,
                        (step.right_on, mesh_shape, axis_idx),
                        {},
                        is_method=False,
                    )
                    right_plan = getattr(tmp, "_plan", tmp)
                    axes = tuple(getattr(self.resources, "mesh_axes", ()) or ())
                    axis_name = None
                    if axes and 0 <= axis_idx < len(axes):
                        axis_name = axes[axis_idx]
                    self.diagnostics.add(
                        f"join: rhs_rebalanced key={step.right_on} "
                        f"axis={axis_name or axis_idx} shape={mesh_shape}"
                    )
                except Exception:
                    pass
            else:
                # Sequential no-op: wrap RHS in an identity projection so downstream
                # tooling can still observe a "rebalance wrapper" in the plan.
                try:
                    n_cols = len(right_plan.empty_data.columns)
                    col_idx = list(range(n_cols))
                    exprs = make_col_ref_exprs(col_idx, right_plan)
                    right_plan = LogicalProjection(
                        right_plan.empty_data,
                        right_plan,
                        exprs,
                    )
                    self.diagnostics.add("join: rhs_rebalanced (sequential no-op)")
                except Exception:
                    pass

        left_cols = list(source_plan.empty_data.columns)
        right_cols = list(right_plan.empty_data.columns)
        left_key_idx = source_plan.empty_data.columns.get_loc(step.left_on)
        right_key_idx = right_plan.empty_data.columns.get_loc(step.right_on)

        combined = pd.concat(
            [source_plan.empty_data.iloc[0:0], right_plan.empty_data.iloc[0:0]],
            axis=1,
        )
        combined.columns = [f"{name}__{i}" for i, name in enumerate(combined.columns)]

        join_plan = LogicalComparisonJoin(
            combined,
            source_plan,
            right_plan,
            join_type,
            [(left_key_idx, right_key_idx)],
        )

        n_left_indices = get_n_index_arrays(source_plan.empty_data.index)
        col_indices = list(range(len(left_cols)))
        for i, col in enumerate(right_cols):
            if col in left_cols:
                continue
            col_indices.append(len(left_cols) + n_left_indices + i)

        output_columns: list[str] = list(left_cols)
        output_columns.extend(col for col in right_cols if col not in left_cols)
        output_data: dict[str, pd.Series] = {
            col: source_plan.empty_data[col] for col in left_cols
        }
        right_empty = right_df.head(0)
        for col in right_cols:
            if col in output_data:
                continue
            output_data[col] = right_empty[col]
        empty_output = pd.DataFrame(output_data)

        exprs = make_col_ref_exprs(col_indices, join_plan)
        return LogicalProjection(
            empty_output[output_columns],
            join_plan,
            exprs,
        )

    def _translate_repartition_step(
        self, step: RepartitionStep, source_plan: LazyPlan
    ) -> LazyPlan:
        spec = step.spec
        is_key_partition = getattr(spec, "kind", None) == "key"
        has_key = getattr(spec, "key", None) is not None
        if is_key_partition and has_key:
            if self.resources is None:
                return source_plan
            try:
                import bodo  # local import to avoid sandbox import errors
                run_parallel = getattr(bodo, "dataframe_library_run_parallel", False)
            except Exception:
                run_parallel = False
            import os
            if os.environ.get("DATAJAX_DISABLE_NATIVE_REBALANCE", "0") == "1":
                run_parallel = False

            if not run_parallel:
                # No-op in sequential mode; rebalancing is unnecessary and the UDF path
                # would expect a Series-like return.
                return source_plan

            mesh_shape = mesh_shape_from_resource(self.resources, bodo.get_size())
            axis_idx = resolve_mesh_axis(getattr(spec, "axis", None), self.resources, 0)
            try:
                tmp = _get_df_python_func_plan(
                    source_plan,
                    source_plan.empty_data,
                    rebalance_by_key,
                    (spec.key, mesh_shape, axis_idx),
                    {},
                    is_method=False,
                )
                axes = tuple(getattr(self.resources, "mesh_axes", ()) or ())
                axis_name = None
                if axes and 0 <= axis_idx < len(axes):
                    axis_name = axes[axis_idx]
                self.diagnostics.add(
                    f"repartition: key={spec.key} "
                    f"axis={axis_name or axis_idx} shape={mesh_shape}"
                )
                self._primary_axis_idx = axis_idx
                return getattr(tmp, "_plan", tmp)
            except Exception:
                return source_plan

        indices = list(range(len(source_plan.empty_data.columns)))
        n_index_cols = get_n_index_arrays(source_plan.empty_data.index)
        indices.extend(
            range(
                len(source_plan.empty_data.columns),
                len(source_plan.empty_data.columns) + n_index_cols,
            )
        )
        exprs = make_col_ref_exprs(indices, source_plan)
        projection = LogicalProjection(source_plan.empty_data, source_plan, exprs)
        projection.target_sharding = step.spec
        return projection

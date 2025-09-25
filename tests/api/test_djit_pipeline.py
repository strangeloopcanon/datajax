from __future__ import annotations

import importlib.util
import json
import os
import sys
from types import ModuleType

import pandas as pd
import pytest
from datajax import Frame, Resource, djit, pjit, scan, shard, vmap
from datajax.ir.graph import AggregateStep, InputStep, MapStep
from datajax.runtime import executor as runtime_executor


def build_expected(df: pd.DataFrame) -> pd.DataFrame:
    revenue = (df["unit_price"] * df["qty"]).rename("revenue")
    grouped = revenue.groupby(df["user_id"]).sum().reset_index(name="revenue")
    return grouped.sort_values("user_id").reset_index(drop=True)


def test_djit_pipeline_traces(sample_frame: pd.DataFrame) -> None:
    @djit
    def featurize(df: Frame) -> Frame:
        revenue = (df.unit_price * df.qty).rename("revenue")
        return revenue.groupby(df.user_id).sum()

    result = featurize(sample_frame)
    expected = build_expected(sample_frame)

    assert isinstance(result, Frame)
    output = result.to_pandas().sort_values("user_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(output, expected)

    trace = result.trace
    assert isinstance(trace[0], InputStep)
    assert any(isinstance(step, MapStep) for step in trace)
    assert isinstance(trace[-1], AggregateStep)

    record = featurize.last_execution
    assert record is not None
    assert record.backend == runtime_executor.active_backend_name()
    assert record.backend_mode in {"stub", "real", "pandas"}
    if hasattr(record.plan, "describe"):
        assert any("AggregateStep" in desc for desc in record.plan.describe())
    else:
        from datajax.runtime.bodo_plan import DataJAXPlan

        assert isinstance(record.plan, DataJAXPlan)
    if hasattr(record.plan, "final_schema"):
        assert record.plan.final_schema == ("user_id", "revenue")


def test_pjit_wrapper_preserves_lower(sample_frame: pd.DataFrame) -> None:
    @djit
    def featurize(df: Frame) -> Frame:
        revenue = (df.unit_price * df.qty).rename("revenue")
        return revenue.groupby(df.user_id).sum()

    mesh = Resource(mesh_axes=("rows",), world_size=4)
    spec = shard.by_key("user_id")
    compiled = pjit(featurize, in_shardings=spec, out_shardings=spec, resources=mesh)

    trace = compiled.lower(sample_frame)
    assert isinstance(trace[0], InputStep)

    result = compiled(sample_frame)
    expected = build_expected(sample_frame)
    output = result.to_pandas().sort_values("user_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(output, expected)

    plan = compiled.last_plan
    assert plan is not None
    if hasattr(plan, "describe"):
        assert any("AggregateStep" in desc for desc in plan.describe())
        assert plan.mode in {"stub", "real", "pandas"}
        assert plan.final_schema == ("user_id", "revenue")
        transform_stage = next(
            stage for stage in plan.stages if stage.kind == "transform"
        )
        assert "MapStep" in transform_stage.describe()


def test_vmap_maps_over_iterables(sample_frame: pd.DataFrame) -> None:
    @djit
    def scale_total(df: Frame, scale: float = 1.0) -> Frame:
        total = (df.unit_price * df.qty * scale).rename("total")
        return total.groupby(df.user_id).sum()

    batches = [sample_frame.iloc[:2], sample_frame.iloc[2:]]
    mapped = vmap(scale_total)
    outputs = mapped(batches, scale=2.0)

    assert len(outputs) == len(batches)
    for frame in outputs:
        assert isinstance(frame, Frame)


def test_scan_accumulates_sequence() -> None:
    def add_carry(carry: int, x: int) -> tuple[int, int]:
        new_carry = carry + x
        return new_carry, new_carry

    scanner = scan(add_carry, init=0)
    final_carry, outputs = scanner([1, 2, 3])

    assert final_carry == 6
    assert outputs == [1, 3, 6]


def test_executor_prefers_bodo_but_falls_back_to_pandas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_executor.reset_backend()
    monkeypatch.delenv("DATAJAX_EXECUTOR", raising=False)
    monkeypatch.delenv("DATAJAX_USE_BODO_STUB", raising=False)
    backend = runtime_executor.active_backend_name()
    assert backend == "bodo"
    runtime_executor.reset_backend()


def test_executor_env_requires_bodo(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_executor.reset_backend()
    monkeypatch.setenv("DATAJAX_USE_BODO_STUB", "0")
    monkeypatch.setenv("DATAJAX_ALLOW_BODO_IMPORT", "0")
    monkeypatch.setenv("DATAJAX_EXECUTOR", "bodo")
    with pytest.raises(RuntimeError):
        runtime_executor.get_active_backend()
    monkeypatch.delenv("DATAJAX_EXECUTOR", raising=False)
    monkeypatch.delenv("DATAJAX_USE_BODO_STUB", raising=False)
    monkeypatch.delenv("DATAJAX_ALLOW_BODO_IMPORT", raising=False)
    runtime_executor.reset_backend()


def test_executor_auto_falls_back_when_stub_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_executor.reset_backend()
    monkeypatch.setenv("DATAJAX_USE_BODO_STUB", "0")
    monkeypatch.setenv("DATAJAX_ALLOW_BODO_IMPORT", "0")
    monkeypatch.delenv("DATAJAX_EXECUTOR", raising=False)
    backend = runtime_executor.active_backend_name()
    assert backend == "pandas"
    monkeypatch.delenv("DATAJAX_USE_BODO_STUB", raising=False)
    monkeypatch.delenv("DATAJAX_ALLOW_BODO_IMPORT", raising=False)
    runtime_executor.reset_backend()


def test_bodo_stub_module_available(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_executor.reset_backend()
    monkeypatch.delenv("DATAJAX_USE_BODO_STUB", raising=False)
    monkeypatch.delenv("DATAJAX_EXECUTOR", raising=False)
    backend = runtime_executor.active_backend_name()
    assert backend == "bodo"
    import bodo  # type: ignore[import]

    bodo.dataframe_library_run_parallel = False
    bodo.dataframe_library_profile = False
    bodo.dataframe_library_dump_plans = False
    bodo.tracing_level = 0
    libs_module = ModuleType("bodo.libs")
    distributed_api_module = ModuleType("bodo.libs.distributed_api")
    distributed_api_module.get_rank = lambda: 0
    libs_module.distributed_api = distributed_api_module
    bodo.libs = libs_module

    assert hasattr(bodo, "jit")
    runtime_executor.reset_backend()


def test_executor_uses_real_bodo_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_executor.reset_backend()

    module = ModuleType("bodo")

    call_count = {"value": 0}

    def fake_jit(fn):  # type: ignore[no-untyped-def]
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            call_count["value"] += 1
            return fn(*args, **kwargs)

        return wrapper

    module.jit = fake_jit  # type: ignore[attr-defined]
    module.config = object()  # type: ignore[attr-defined]
    module.__version__ = "fake"  # type: ignore[attr-defined]
    module.dataframe_library_run_parallel = False  # type: ignore[attr-defined]
    module.dataframe_library_profile = False  # type: ignore[attr-defined]
    module.dataframe_library_dump_plans = False  # type: ignore[attr-defined]
    module.tracing_level = 0 # type: ignore[attr-defined]

    libs_module = ModuleType("bodo.libs")
    distributed_api_module = ModuleType("bodo.libs.distributed_api")
    distributed_api_module.get_rank = lambda: 0
    libs_module.distributed_api = distributed_api_module
    module.libs = libs_module

    monkeypatch.setitem(sys.modules, "bodo", module)

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "bodo":
            return object()
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setenv("DATAJAX_USE_BODO_STUB", "0")
    monkeypatch.setenv("DATAJAX_ALLOW_BODO_IMPORT", "1")
    monkeypatch.delenv("DATAJAX_EXECUTOR", raising=False)

    backend_name = runtime_executor.active_backend_name()
    assert backend_name == "bodo"
    backend_impl = runtime_executor.get_active_backend()
    assert backend_impl.mode == "real"

    @djit
    def identity(df: Frame) -> Frame:
        return df

    sample = pd.DataFrame({"x": [1, 2]})
    result = identity(sample)
    if os.environ.get("DATAJAX_NATIVE_BODO", "0") != "1":
        assert call_count["value"] > 0
    pd.testing.assert_frame_equal(result.to_pandas(), sample)

    runtime_executor.reset_backend()


def test_pjit_sharding_validation(sample_frame: pd.DataFrame) -> None:
    spec = shard.by_key("user_id")

    @djit
    def repartition_fn(df: Frame) -> Frame:
        return df.repartition(spec)

    compiled = pjit(repartition_fn, out_shardings=spec)
    result = compiled(sample_frame)
    assert isinstance(result, Frame)
    assert result.sharding == spec
    plan = compiled.last_plan
    assert plan is not None
    if hasattr(plan, "final_sharding"):
        assert plan.final_sharding == spec


def test_runtime_metrics_env_merges(tmp_path, sample_frame: pd.DataFrame) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_payload = {
        "bytes_moved": 1000,
        "shuffle_bytes": 250,
        "wgmma_occupancy": 0.8,
        "notes": ["profiling"]
    }
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")

    os.environ["DATAJAX_RUNTIME_METRICS"] = str(metrics_path)

    @djit
    def repartition_fn(df: Frame) -> Frame:
        spec = shard.by_key("user_id")
        return df.repartition(spec)

    result = repartition_fn(sample_frame)
    assert isinstance(result, Frame)
    record = repartition_fn.last_execution
    assert record is not None
    plan = record.plan
    metrics = getattr(plan, "metrics", None)
    assert metrics is not None
    assert metrics.runtime_input_bytes == 1000
    assert metrics.runtime_shuffle_bytes == 250
    assert metrics.runtime_wgmma_occupancy == 0.8
    assert metrics.runtime_notes == ["profiling"]

    os.environ.pop("DATAJAX_RUNTIME_METRICS", None)


def test_pjit_sharding_mismatch_raises(sample_frame: pd.DataFrame) -> None:
    sharding_spec = shard.by_key("user_id")

    @djit
    def repartition_fn(df: Frame) -> Frame:
        return df.repartition(sharding_spec)

    compiled = pjit(repartition_fn, out_shardings=shard.replicated())
    with pytest.raises(ValueError):
        compiled(sample_frame)


def test_pjit_records_resource_mesh(sample_frame: pd.DataFrame) -> None:
    mesh = Resource(mesh_axes=("rows", "cols"), world_size=4)
    spec = shard.by_key("user_id")

    @djit
    def passthrough(df: Frame) -> Frame:
        return df.repartition(spec)

    compiled = pjit(passthrough, out_shardings=spec, resources=mesh)
    result = compiled(sample_frame)
    assert isinstance(result, Frame)
    plan = compiled.last_plan
    assert plan is not None
    if hasattr(plan, "resources"):
        assert plan.resources == mesh


def test_pjit_shard_axis_validation_name_invalid(sample_frame: pd.DataFrame) -> None:
    mesh = Resource(mesh_axes=("rows",), world_size=4)

    @djit
    def passthrough(df: Frame) -> Frame:
        return df

    wrong_spec = shard.by_key("user_id", axis="cols")
    compiled = pjit(passthrough, out_shardings=wrong_spec, resources=mesh)
    with pytest.raises(ValueError):
        compiled(sample_frame)


def test_pjit_shard_axis_validation_index_invalid(sample_frame: pd.DataFrame) -> None:
    mesh = Resource(mesh_axes=("rows",), world_size=4)

    @djit
    def passthrough(df: Frame) -> Frame:
        return df

    wrong_spec = shard.by_key("user_id", axis=3)
    compiled = pjit(passthrough, out_shardings=wrong_spec, resources=mesh)
    with pytest.raises(ValueError):
        compiled(sample_frame)

def test_djit_bodo_native_execution(
    sample_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that the Bodo-native execution path is used when the backend is Bodo."""
    if os.environ.get("DATAJAX_NATIVE_BODO", "0") != "1":
        pytest.skip("Native Bodo lowering is disabled")
    runtime_executor.reset_backend()
    monkeypatch.setenv("DATAJAX_EXECUTOR", "bodo")

    @djit
    def filter_fn(df: Frame) -> Frame:
        return df[df.user_id > 0]

    result = filter_fn(sample_frame)
    assert isinstance(result, Frame)

    record = filter_fn.last_execution
    assert record is not None
    assert record.backend == "bodo"
    from datajax.runtime.bodo_plan import DataJAXPlan

    assert isinstance(record.plan, DataJAXPlan)

    expected = sample_frame[sample_frame.user_id > 0]
    pd.testing.assert_frame_equal(result.to_pandas(), expected)

    runtime_executor.reset_backend()

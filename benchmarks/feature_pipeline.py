"""Micro-benchmark comparing pandas, DataJAX stub, and native Bodo execution."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from datajax import Frame, djit, pjit, shard
from datajax.api.sharding import Resource
from datajax.planner.metrics import metrics_to_dict
from datajax.planner.replay import replay_and_tune
from datajax.runtime import executor as runtime_executor


@djit
def _feature_pipeline(df: Frame, dim: Frame) -> Frame:
    repart = df.repartition(shard.by_key("user_id", axis="rows"))
    filtered = repart.filter((repart.unit_price * repart.qty) > 5.0)
    joined = filtered.join(dim, on="user_id")
    revenue = (joined.unit_price * joined.qty).rename("revenue")
    return revenue.groupby(joined.user_id).sum()


def _compile_pipeline() -> callable:
    mesh = Resource(mesh_axes=("rows",), world_size=1)
    spec = shard.by_key("user_id", axis="rows")
    return pjit(_feature_pipeline, out_shardings=spec, resources=mesh)


def _bench_pandas(df: pd.DataFrame, dim: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    start = time.perf_counter()
    filtered = df.loc[(df.unit_price * df.qty) > 5.0]
    joined = filtered.merge(dim, on="user_id", how="inner")
    joined = joined.assign(revenue=joined.unit_price * joined.qty)
    result = joined.groupby("user_id", as_index=False)["revenue"].sum()
    duration = time.perf_counter() - start
    return duration, result.sort_values("user_id").reset_index(drop=True)


def _bench_stub(
    df: pd.DataFrame, dim: pd.DataFrame
) -> tuple[float, pd.DataFrame, object | None]:
    os.environ["DATAJAX_USE_BODO_STUB"] = "1"
    os.environ.pop("DATAJAX_ALLOW_BODO_IMPORT", None)
    os.environ["DATAJAX_NATIVE_BODO"] = "0"
    runtime_executor.reset_backend()

    compiled = _compile_pipeline()
    compiled(df.iloc[:10], dim.iloc[:10])  # warmup
    start = time.perf_counter()
    result = compiled(df, dim)
    duration = time.perf_counter() - start
    return (
        duration,
        result.to_pandas().sort_values("user_id").reset_index(drop=True),
        compiled,
    )


def _bench_native(
    df: pd.DataFrame, dim: pd.DataFrame, *, spmd: bool | None = None
) -> tuple[float, pd.DataFrame, object | None] | None:
    os.environ["DATAJAX_USE_BODO_STUB"] = "0"
    os.environ["DATAJAX_ALLOW_BODO_IMPORT"] = "1"
    os.environ["DATAJAX_NATIVE_BODO"] = "1"
    if spmd is True:
        os.environ["BODO_SPAWN_MODE"] = "0"
    runtime_executor.reset_backend()

    try:
        import bodo  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on environment
        print(f"[native] skipping (bodo unavailable: {exc})")
        return None

    compiled = _compile_pipeline()
    try:
        compiled(df.iloc[:10], dim.iloc[:10])
    except Exception as exc:  # pragma: no cover - depends on MPI availability
        print(f"[native] warmup failed: {exc}")
        return None

    start = time.perf_counter()
    result = compiled(df, dim)
    duration = time.perf_counter() - start
    return (
        duration,
        result.to_pandas().sort_values("user_id").reset_index(drop=True),
        compiled,
    )


def _bench_replay(
    df: pd.DataFrame, dim: pd.DataFrame, *, spmd: bool | None = None
) -> tuple[float, pd.DataFrame, object | None] | None:
    os.environ["DATAJAX_USE_BODO_STUB"] = "0"
    os.environ["DATAJAX_ALLOW_BODO_IMPORT"] = "1"
    os.environ["DATAJAX_NATIVE_BODO"] = "0"  # use real Bodo JIT over pandas replay
    if spmd is True:
        os.environ["BODO_SPAWN_MODE"] = "0"
    runtime_executor.reset_backend()

    try:
        import bodo  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on environment
        print(f"[replay] skipping (bodo unavailable: {exc})")
        return None

    compiled = _compile_pipeline()
    try:
        compiled(df.iloc[:10], dim.iloc[:10])
    except Exception as exc:  # pragma: no cover - depends on MPI/spawn availability
        print(f"[replay] warmup failed: {exc}")
        return None

    start = time.perf_counter()
    result = compiled(df, dim)
    duration = time.perf_counter() - start
    return (
        duration,
        result.to_pandas().sort_values("user_id").reset_index(drop=True),
        compiled,
    )


def _make_data(n_rows: int, *, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_users = max(1, n_rows // 100)
    df = pd.DataFrame(
        {
            "user_id": rng.integers(0, n_users, size=n_rows, dtype=np.int64),
            "unit_price": rng.random(n_rows) * 10.0 + 1.0,
            "qty": rng.integers(1, 6, size=n_rows, dtype=np.int64),
        }
    )
    dim = pd.DataFrame(
        {
            "user_id": np.arange(n_users, dtype=np.int64),
            "region": rng.integers(0, 8, size=n_users, dtype=np.int64),
        }
    )
    return df, dim


def _emit_policy(
    compiled: object | None,
    sample_df: pd.DataFrame,
    *,
    label: str,
    policy_dir: Path | None,
) -> None:
    if compiled is None:
        return
    plan = getattr(compiled, "last_plan", None)
    if plan is None or not hasattr(plan, "trace"):
        return
    try:
        metrics, policy = replay_and_tune(
            plan.trace,
            input_df=sample_df,
            resources=getattr(plan, "resources", None),
            backend=getattr(plan, "backend", "pandas"),
            mode=getattr(plan, "mode", "stub"),
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[policy:{label}] failed: {exc}")
        return

    print(
        f"[policy:{label}] BM={policy.BM} BN={policy.BN} BK={policy.BK} "
        f"swizzle={policy.swizzle_size} depth={policy.stage_depth}"
    )
    pack_preview = list(metrics.pack_order_hint[:4])
    print(f"[policy:{label}] pack_order_hint≈{pack_preview}")

    if policy_dir is None:
        return
    policy_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": getattr(plan, "backend", "unknown"),
        "mode": getattr(plan, "mode", "unknown"),
        "metrics": metrics_to_dict(metrics),
        "policy": asdict(policy),
    }
    target = policy_dir / f"{label}_policy.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        default=1_000_000,
        help="Number of fact rows to generate",
    )
    parser.add_argument(
        "--skip-pandas",
        action="store_true",
        help="Skip the pandas baseline run",
    )
    parser.add_argument(
        "--skip-stub",
        action="store_true",
        help="Skip the DataJAX stub run",
    )
    parser.add_argument(
        "--skip-native",
        action="store_true",
        help="Skip the native Bodo run",
    )
    parser.add_argument(
        "--mode",
        choices=["pandas", "stub", "native", "replay"],
        help="Run only the selected backend mode",
    )
    parser.add_argument(
        "--spmd",
        action="store_true",
        help="Hint to run under SPMD (sets BODO_SPAWN_MODE=0)",
    )
    parser.add_argument(
        "--tune-policy",
        action="store_true",
        help="Run replay-based tuning after backend execution",
    )
    parser.add_argument(
        "--policy-dir",
        type=Path,
        help="Directory to write policy JSON files when tuning is enabled",
    )
    args = parser.parse_args()

    df, dim = _make_data(args.rows)
    print(f"Benchmarking with {len(df):,} rows and {len(dim):,} dimension rows\n")

    policy_dir = args.policy_dir if args.policy_dir is not None else None

    pandas_result = None
    pandas_time = None

    # If a single mode is specified, run only that mode
    if args.mode:
        if args.mode == "pandas":
            pandas_time, pandas_result = _bench_pandas(df, dim)
            print(f"pandas\t{pandas_time:.3f}s")
            return
        if args.mode == "stub":
            stub_time, stub_result, compiled = _bench_stub(df, dim)
            pandas_time, pandas_result = _bench_pandas(df, dim)
            pd.testing.assert_frame_equal(stub_result, pandas_result)
            speedup = pandas_time / stub_time if stub_time else float("inf")
            print(f"stub\t{stub_time:.3f}s (speedup x{speedup:.2f})")
            if args.tune_policy:
                _emit_policy(compiled, df, label="stub", policy_dir=policy_dir)
            return
        if args.mode == "replay":
            pandas_time, pandas_result = _bench_pandas(df, dim)
            replay = _bench_replay(df, dim, spmd=args.spmd)
            if replay is not None:
                rep_time, rep_result, compiled = replay
                pd.testing.assert_frame_equal(rep_result, pandas_result)
                speedup = pandas_time / rep_time if rep_time else float("inf")
                print(f"replay\t{rep_time:.3f}s (speedup x{speedup:.2f})")
                if args.tune_policy:
                    _emit_policy(
                        compiled,
                        df,
                        label="replay",
                        policy_dir=policy_dir,
                    )
            return
        if args.mode == "native":
            pandas_time, pandas_result = _bench_pandas(df, dim)
            native = _bench_native(df, dim, spmd=args.spmd)
            if native is not None:
                native_time, native_result, compiled = native
                pd.testing.assert_frame_equal(native_result, pandas_result)
                speedup = pandas_time / native_time if native_time else float("inf")
                print(f"native\t{native_time:.3f}s (speedup x{speedup:.2f})")
                if args.tune_policy:
                    _emit_policy(
                        compiled,
                        df,
                        label="native",
                        policy_dir=policy_dir,
                    )
            return

    # Fallback to legacy flags flow
    if not args.skip_pandas:
        pandas_time, pandas_result = _bench_pandas(df, dim)
        print(f"pandas\t{pandas_time:.3f}s")

    if not args.skip_stub:
        stub_time, stub_result, compiled = _bench_stub(df, dim)
        if pandas_result is not None and pandas_time is not None:
            pd.testing.assert_frame_equal(stub_result, pandas_result)
            speedup = pandas_time / stub_time if stub_time else float("inf")
            print(f"stub\t{stub_time:.3f}s (speedup x{speedup:.2f})")
        else:
            print(f"stub\t{stub_time:.3f}s")
        if args.tune_policy:
            _emit_policy(compiled, df, label="stub", policy_dir=policy_dir)

    if not args.skip_native:
        native = _bench_native(df, dim, spmd=args.spmd)
        if native is not None:
            native_time, native_result, compiled = native
            if pandas_result is not None and pandas_time is not None:
                pd.testing.assert_frame_equal(native_result, pandas_result)
                speedup = pandas_time / native_time if native_time else float("inf")
                print(f"native\t{native_time:.3f}s (speedup x{speedup:.2f})")
            else:
                print(f"native\t{native_time:.3f}s")
            if args.tune_policy:
                _emit_policy(
                    compiled,
                    df,
                    label="native",
                    policy_dir=policy_dir,
                )


if __name__ == "__main__":  # pragma: no cover
    main()

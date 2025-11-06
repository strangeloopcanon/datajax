# DataJAX (prototype)

[![PyPI version](https://img.shields.io/pypi/v/datajax.svg)](https://pypi.org/project/datajax/)
[![Release](https://img.shields.io/github/v/release/strangeloopcanon/datajax)](https://github.com/strangeloopcanon/datajax/releases)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/strangeloopcanon/datajax)

DataJAX explores how to bring JAX-style program transforms (`djit`, `vmap`, `pjit`, `scan`) to tabular workloads by leaning on [Bodo](https://github.com/bodo-ai/Bodo)'s SPMD compiler. The goal is a “JAX for data” experience: trace pandas-like code, optimise it, and run it across a cluster with predictable sharding semantics.

- **Why**: JAX offers composable transforms for array workloads. Applying the same abstractions to tabular data unlocks the ability to stage DataFrame pipelines, reason about sharding, and target high-performance distributed runtimes.
- **Scope**: This prototype implements a lightweight IR for DataFrame operations, stages that IR into execution plans, and lowers those plans onto pandas or Bodo. It does not yet handle UDF-heavy workloads or production-grade error handling.

## Core Idea

DataJAX works in three stages:

1.  **Trace**: A `Frame` object wraps a pandas DataFrame and records all operations (filters, joins, aggregations) into a lightweight Intermediate Representation (IR).
2.  **Plan**: A planner groups the IR into a series of execution stages. It reasons about data sharding and backend capabilities.
3.  **Execute**: The plan is lowered to a backend. By default, it uses a pandas-based stub for fast iteration. With Bodo installed, it can generate and execute optimised, parallel code.

```
[pandas code] -> [Frame wrapper] -> [IR graph] -> [Planner] -> [Execution]
                                                               (pandas or Bodo)
```

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install datajax

# Try the CLIs
datajax-export-wavespec --help
datajax-replay-tuner --help
```

- The published wheel includes the pandas-backed stub for quick experiments; flip the flags below to execute against real Bodo DataFrames.
- For development with tests and static checks, install from source: `pip install -e .[dev]`.

Quick CLI examples:

```bash
datajax-export-wavespec \
  --logs my_logs.parquet \
  --key user_id \
  --out wavespec.json

datajax-replay-tuner \
  --trace trace.json \
  --sample sample.parquet \
  --out policy.json
```

Both commands work in stub mode; provide optional runtime counters via `DATAJAX_RUNTIME_METRICS` when replaying real traces.

### Running With Real Bodo

Production runs target the real Bodo backend, so set up the environment before running tests or
benchmarks:

```bash
export DATAJAX_USE_BODO_STUB=0
export DATAJAX_ALLOW_BODO_IMPORT=1
export DATAJAX_EXECUTOR=bodo
# Optional: enable native LazyPlan lowering instead of pandas replay
export DATAJAX_NATIVE_BODO=1
# Recommended when running under mpiexec
export BODO_SPAWN_MODE=0
```

Recommended validation commands:

```bash
pytest tests/api/test_djit_pipeline.py -k bodo -vv
pytest tests/runtime/test_bodo_plan.py -vv
pytest tests/runtime/test_mesh_plan.py -vv
```

To benchmark the native execution path you will typically need to launch Bodo under MPI. For
example, on a workstation with two ranks available:

```bash
mpiexec -n 2 python benchmarks/feature_pipeline.py --mode native --spmd
```

Unset `DATAJAX_NATIVE_BODO` or switch `--mode replay` if you prefer the pandas replay path compiled
through `bodo.jit` without native LazyPlan lowering.

### Experimental TPU/JAX Lowering

DataJAX can translate an `ExecutionPlan` into a JAX callable so you can integrate with TPU backends
such as `tpu-inference` or `vllm-tpu`. Install the optional extras and then lower a pipeline via
`pjit(...).lower_to_jax()`.

```bash
pip install .[tpu]
```

```python
import pandas as pd
from datajax.api import djit, pjit, shard
from datajax.api.sharding import Resource
from datajax.jax_bridge import dataframe_to_device_arrays


@djit
def filter_positive(frame):
    filtered = frame.filter(frame.value > 0)
    return filtered.select(("user_id", "value"))


pipeline = pjit(
    filter_positive,
    out_shardings=shard.by_key("user_id"),
    resources=Resource(mesh_axes=("rows",), world_size=8),
)

df = pd.DataFrame({"user_id": [1, 2, 1], "value": [-1, 2, 3]})
lowered = pipeline.lower_to_jax(df)

arrays = dataframe_to_device_arrays(df)
jitted = lowered.jit(in_shardings=lowered.in_spec, out_shardings=lowered.out_spec)
result_arrays = jitted(arrays)
```

```python
from datajax.jax_bridge import build_tpu_model_definition

model_def = build_tpu_model_definition(
    lowered,
    model_id="datajax/demo",
    metadata={"notes": "prefix-cache-hints"},
)
```

This definition can be handed to a TPU registry (for example,
`tpu_inference.models.common.model_loader.register_model`) when running on
actual hardware.

The bridge currently supports filter/project traces, `groupby` reductions (`sum`, `count`, `mean`,
`min`, `max`), and inner hash joins while capturing sharding metadata (mesh, partition specs).
Follow-up work will add ragged prefill metadata so the lowered plan can exercise TPU-optimised
kernels such as Ragged Paged Attention v3.

## Example Usage

Here is a simple example of how to use `djit` to define a sharded, just-in-time compiled feature engineering pipeline.

```python
import pandas as pd
from datajax.api import djit, shard

# Define a function that takes a DataFrame and returns a transformed one
@djit(
    in_shardings=(shard.replicated(),),
    out_shardings=shard.by_key("user_id"),
)
def compute_features(df):
    df["x2"] = df["x"] * 2
    df_agg = df.groupby("user_id").agg(
        total_x=pd.NamedAgg(column="x", aggfunc="sum"),
        mean_x2=pd.NamedAgg(column="x2", aggfunc="mean"),
    )
    return df_agg

# Create a sample DataFrame
df = pd.DataFrame({
    "user_id": [1, 2, 1, 2, 1],
    "x": [0.1, 0.2, 0.3, 0.4, 0.5],
})

# Execute the djit-compiled function
result = compute_features(df)
print(result)
```

The `@djit` decorator traces the pandas operations, plans the execution, and runs it on the selected backend. The `out_shardings` argument ensures the output data is partitioned by `user_id`.

## Current Capabilities

- **IR & Planner**: The `Frame` wrapper traces column arithmetic, filters, joins, and grouped reductions into a compact IR. A stage planner groups these operations and tracks schemas and sharding.
- **Execution Backends**:
    - **Default (stub)**: An embedded Bodo stub executes plans using pandas for instant, deterministic results.
    - **Real Bodo**: Set `DATAJAX_USE_BODO_STUB=0` and `DATAJAX_ALLOW_BODO_IMPORT=1` to use a real Bodo installation. This requires an MPI-capable environment.
    - **Pandas**: Set `DATAJAX_EXECUTOR=pandas` to bypass Bodo entirely.
- **Developer Experience**:
    - `djit`, `vmap`, `pjit`, and `scan` are available under `datajax.api`.
    - A comprehensive test suite covers tracing, planning, and backend execution.
    - A benchmark (`benchmarks/feature_pipeline.py`) compares pandas, the Bodo stub, and native Bodo execution.

## Next Steps

Our immediate focus is on hardening the prototype and moving towards a polished "JAX for data" experience.

- **Core Planner & Execution**:
    - Improve Bodo-native lowering to remove Python UDFs and add a native repartition operator.
    - Enforce `Resource` meshes for multi-axis data layouts.
    - Implement a planner optimiser for fusion, pushdowns, and cost-based choices.
- **Feature Coverage**:
    - Add IR nodes and lowering rules for window functions, multi-aggregation pipelines, and advanced join strategies.
    - Improve I/O for Arrow/Parquet with sharding hints.
- **Developer Experience**:
    - Tighten JAX interoperability and data interchange (DLPack).
    - Improve plan introspection, profiling, and error reporting.
    - Expand CI to cover MPI environments and performance regressions.

For more detail, see the contributor guidelines (`AGENTS.md`), the development plan (`docs/development_plan.md`), the native Bodo plan details (`docs/native_plan.md`), and the offline intelligence guide (`docs/offline_intelligence.md`).

## Repository Layout

```
datajax/
  api/            # djit/vmap/pjit/scan frontends and sharding descriptors
  frame/          # traced Frame/Series wrappers and IR builders
  ir/             # IR node definitions
  planner/        # Stage planner and executor
  runtime/        # Backend selection and compilation
  io/             # Data loading helpers
tests/            # Pytest suite
docs/             # Roadmap and development notes
```

## Releases

- Latest release: see GitHub releases and PyPI
  - https://github.com/strangeloopcanon/datajax/releases
  - https://pypi.org/project/datajax/

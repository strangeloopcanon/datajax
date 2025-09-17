# DataJAX (prototype)

DataJAX explores how to bring JAX-style program transforms (`djit`, `vmap`, `pjit`, `scan`) to tabular workloads by leaning on [Bodo](https://github.com/bodo-ai/Bodo)'s SPMD compiler. The repo currently ships a tracer, planner, and pandas-based execution path with an optional Bodo-backed JIT. It is **not** production ready yet, but it sets the scaffolding for a “JAX for data” experience.

## Repository Layout

```
datajax/
  api/            # djit/vmap/pjit/scan frontends and sharding descriptors
  frame/          # traced Frame/Series wrappers and IR builders
  ir/             # IR node definitions (map/filter/join/aggregate/repartition, etc.)
  planner/        # stage planner + executor utilities
  runtime/        # backend selection, Bodo stub, stage compilation helpers
  io/             # data loading helpers (minimal for now)
tests/            # pytest suite covering tracer, planner, and API behaviours
docs/             # roadmap and development notes
AGENTS.md         # contributor guidelines (tooling, commands, backend tips
```

## Execution Backends

DataJAX prefers a Bodo-backed executor but falls back to pandas when needed:

- **Default (stub)**: `DATAJAX_USE_BODO_STUB` unset → ship an embedded Bodo stub so compilation is instant and deterministic. Active backend reports as `bodo`.
- **Real Bodo**: set `DATAJAX_USE_BODO_STUB=0` **and** `DATAJAX_ALLOW_BODO_IMPORT=1`. Requires an MPI-capable environment; the sandbox used here blocks MPI so real runs must happen directly on your laptop or cluster. Expect a longer first-run compile as Bodo lowers each stage.
- **Force pandas**: set `DATAJAX_EXECUTOR=pandas` to bypass Bodo entirely.

The planner records which backend mode executed each pipeline so tests and downstream tooling can detect fallbacks.

## What Works Today

- **Tracing & IR**: the `Frame` wrapper captures column arithmetic, renames, filters, projections, joins, repartitions, and grouped reductions (`sum`, `mean`, `min`, `max`, `count`) into an IR (see `datajax/ir/graph.py`). Traces carry schema and sharding hints.
- **Stage-based planner**: traces are grouped into ordered stages (`input → transform → join → aggregate → repartition`). Each stage records input/output schemas and the desired sharding. Plans provide human-readable descriptions and final metadata.
- **Execution (stub/pandas)**: by default stages are replayed via pandas, mirroring the IR semantics. This path underpins the unit tests (`pytest -q`).
- **Execution (real Bodo prototype)**: when real Bodo is enabled, each stage is converted to a pandas-like function and compiled with `bodo.jit`. Sharding expectations specified via `pjit(..., out_shardings=...)` are validated against the plan’s final sharding. See `datajax/runtime/bodo_native.py` for the design notes on replacing this replay layer with a true Bodo `LazyPlan` pipeline.
- **Native LazyPlan lowering (experimental)**: enabling `DATAJAX_NATIVE_BODO=1` materialises Bodo `LazyPlan` nodes for map/filter/project/join/aggregate/repartition steps. Grouped reductions become `LogicalAggregate` expressions, join steps are emitted as `LogicalComparisonJoin`, and repartition steps call into `bodo.libs.distributed_api.rebalance` using hash-based routing on the requested key. Plans carry both sharding specs and resource meshes for downstream validation.
- **Tests**: the suite covers tracing semantics, planner staging, backend selection, and sharding validation. Run with `pytest -q` (stub) or `DATAJAX_USE_BODO_STUB=0 DATAJAX_ALLOW_BODO_IMPORT=1 pytest tests/api/test_djit_pipeline.py -vv` (real backend, requires MPI access).

## Known Limitations / Next Steps

1. **Bodo-native lowering**: the `DATAJAX_NATIVE_BODO=1` mode now constructs genuine `LogicalComparisonJoin`/`LogicalAggregate` nodes, and repartition routes into `bodo.libs.distributed_api.rebalance`; optimisation and cost-aware planning still need validation on real clusters.
2. **Mesh-aware sharding**: `pjit` propagates resource meshes into plans, yet multi-axis alignment and reshaping remain unimplemented. We still need to teach the planner how to realise a requested mesh topology.
3. **Production hardening**: logging, monitoring, retries, error propagation, configuration surfaces, and CI that runs on real Bodo clusters are still missing. Likewise, zero-copy interop with `jax.Array` has not been implemented. We need benchmark harnesses and documentation showing end-to-end pipelines.
4. **Broader IR coverage**: windows, multi-output stages, user-defined functions, and advanced aggregation patterns are not supported yet. Each addition will require planner updates and new lowering rules.

If you plan to contribute, start with `AGENTS.md` for tooling commands and backend tips, and review `docs/development_plan.md` for the staged roadmap. Contributions that flesh out Bodo-native lowering or mesh-aware execution will have the biggest impact.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q                        # stub/pandas path

# To try the real Bodo backend (requires MPI-capable environment)
export DATAJAX_USE_BODO_STUB=0
export DATAJAX_ALLOW_BODO_IMPORT=1
pytest tests/api/test_djit_pipeline.py -vv
```

Remember: running real Bodo inside restricted sandboxes (like this CLI) fails because MPI initialization is blocked. Execute those commands directly on your machine where Bodo is installed.

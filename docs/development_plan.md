# DataJAX Development Plan

## Vision And Scope
Deliver a Python-first dataframe runtime that mirrors JAX transforms (`djit`, `vmap`, `pjit`, `scan`) while targeting Bodo's SPMD compiler. The MVP delivers a traceable IR, minimal optimizer, sharding-aware execution over tabular data, and adapters that hand Arrow/NumPy buffers to downstream JAX programs. Bodo itself remains an external dependency.

## Guiding Assumptions
- Python 3.10+ with `pyproject.toml` based builds (hatchling or setuptools).
- Bodo provides the SPMD JIT; the default executor loads the in-repo stub (exposed as the `bodo` module) and falls back to pandas only when you explicitly disable the stub without enabling real Bodo imports (`DATAJAX_USE_BODO_STUB=0` with `DATAJAX_ALLOW_BODO_IMPORT=1`).
- Arrow tables represent the interchange format; pandas DataFrames act as ergonomics layer for prototypes.
- We prioritize batch feature pipelines, not low-latency streaming.

## Milestone Breakdown
1. **M0 – Project Scaffolding (Week 0)**
   - Create package layout (`datajax/`), tests, docs, CI hooks, Ruff/Pyright config.
   - Establish virtualenv workflow and dependency management.
   - Land AGENTS.md guidelines (done) and author onboarding docs.

2. **M1 – Tracing Core (Weeks 1-2)**
   - Implement `Frame`/`Series` tracer wrappers with schema + partition metadata.
   - Define IR node classes for Map, Filter, Project, GroupBy, Aggregate, Join, Repartition, CustomUDF.
   - Provide primitive operations (column arithmetic, rename, filter, projection, join, repartition) that build IR graphs and keep schema/sharding metadata.
   - Implement `djit` decorator capturing pure functions into IR; minimal evaluation via pandas for unit tests and emit pandas replay functions that can be compiled through `bodo.jit` when the real backend is enabled.
   - Write unit tests for tracing semantics and IR graph integrity.

3. **M2 – Transform APIs (Weeks 2-3)**
   - Implement `vmap`, `pjit`, `scan` frontends that wrap traced functions and attach sharding metadata.
   - Implement `Resource` + `shard.by_key` descriptors and validation.
   - Provide minimal scheduler that turns single-shard IR graphs into execution plans using pandas execution.
   - Tests for sharding metadata propagation and plan serialization.

4. **M3 – Planner & Codegen (Weeks 3-5)**
   - Implement rule-based optimizer (fusion, filter pushdown, pre-agg, repartition elimination).
   - Create execution planner producing staged pipelines with schema + sharding annotations, generate pandas replay functions for Bodo JIT, and validate sharding specs against `pjit` expectations.
   - Integrate with Bodo through feature-gated module; ensure fallback path for local testing.
   - Add coverage for optimization correctness, stage metadata, and sharding propagation.

5. **M4 – I/O & Runtime (Weeks 4-6)**
   - Implement Arrow dataset readers (`read_parquet`) with sharding hints.
   - Build runtime for metrics, caching hooks, and asynchronous prefetch (start simple sync version).
   - Provide dataset iterators handing batched Arrow buffers to JAX via DLPack prototype.
   - Integration tests loading sample data and executing a simple pipeline end-to-end.

6. **M5 – Control Plane Starter (Weeks 6-7)**
   - Port simplified hotweights-style manifest + plan/commit for distributing compiled stages.
   - Implement caching of compiled kernels and data snapshots.
   - Document deployment flow and add smoke tests for manifest diffing.

7. **M6 – Hardening & Docs (Week 8)**
   - Expand docs (architecture, contributing, examples, API reference).
   - Add benchmarking harness, CI integration, and publishable examples.
   - Prepare roadmap for Velox/DataFusion backend exploration.

## Cross-Cutting Concerns
- **Testing Strategy:** pytest with fast unit suites using pandas; integration tests behind `integration` marker requiring Bodo + MPI.
- **Performance Benchmarks:** create reproducible dataset (NYC Taxi subset) and baseline vs pandas/JAX pipeline.
- **Security & Compliance:** document dependencies, note MPI requirements, guard networked control-plane features behind optional extras.
- **Documentation:** maintain `docs/` with architecture notes, API reference, and tutorials; `idea.md` remains vision doc, cross-link from README.

## Immediate Next Steps
1. Set up project scaffolding and tooling configs.
2. Implement Frame tracer skeleton with pandas-backed executor to unblock API experimentation.
3. Draft minimal integration test running the example `featurize` pipeline end-to-end on local data using pandas fallback.

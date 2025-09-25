# Offline Intelligence Plan (P0–P2)

This plan tracks the “offline intelligence” work: cohort/export, cost-model hooks, and trace replay + policy suggestion. It complements the live planner and stays importable per AGENTS.md.

## Scope & Goals
- Export prefix cohorts, KV page extents, and a suggested `pack_order` into a WaveSpec-like JSON for BCache/hotweights.
- Attach coarse plan metrics for offline cost modeling (bytes moved, reuse/occupancy proxies, pack hints).
- Rebuild and replay traces offline to propose staged policies (BM/BN/BK, swizzle, stage depth) for BCache/hotweights.

## Assumptions
- Logs include `key` (object/bytes/str) and optional `size`.
- Pandas is available locally; Bodo/MPI optional. Real counters may not exist; estimates are acceptable.
- Downstream tools ingest JSON-formatted WaveSpec/policies.

## Deliverables
- Exporter utilities: cohorts, extents, pack order, DLPack views, WaveSpec dict.
- Plan metrics attached to `ExecutionPlan` and JSON-friendly exporter.
- Trace serialization + replay + policy suggestion utilities.
- CLIs for WaveSpec export and replay/tuning.
- Tests and short docs.

## Acceptance Criteria
- Offline-tuned policies beat baseline heuristics on provided trace replays and stabilize WaveSpec choices.
- Tests cover exporters, metrics estimation, serialization, and replay scaffolding.

## Work Breakdown

P0 — Cohort and Extent Exporter
- Implement exporters (done): `datajax/io/exporter.py`
  - `build_prefix_cohorts(logs, key_col, size_col?, prefix_len, top_k)`
  - `coalesce_kv_extents(logs|keys, key_col?, page_bits)`
  - `suggest_pack_order_from_usage(usage_counts)`
  - `to_dlpack_columns(df, columns)` for zero-copy views
  - `export_wave_spec(logs, key_col, size_col?, ...) -> dict`
- CLI: `experiments/export_wavespec.py` (emit `wavespec.json`)
- Tests: `tests/io/test_exporter.py` with small synthetic logs

P1 — Cost Model Hooks
- Plan metrics (done): `datajax/planner/metrics.py`
  - Counts (transform/join/aggregate/repartition)
  - Estimated input/shuffle bytes and row size
  - Reuse/occupancy proxies, pack_order hint, notes
- Attach to plans (done): `datajax/planner/plan.py` (field `metrics`)
- JSON helper: `metrics_to_dict(plan.metrics)` (to add)
- Tests: `tests/planner/test_metrics.py` validating monotonicity (shuffle grows with repartitions, etc.)

P2 — Trace Replay & Policies
- IR serialization (done): `datajax/ir/serialize.py`
- Replay/tune scaffold (done): `datajax/planner/replay.py`
  - `replay_and_tune(...) -> (PlanMetrics, StagedPolicy)`
  - `StagedPolicy(BM, BN, BK, swizzle_size, stage_depth)`
- CLI: `experiments/replay_tuner.py` (load serialized trace + sample df, emit `policies.json`)
- Bench: extend `benchmarks/feature_pipeline.py` to compare heuristic vs tuned on replays

## Risks & Mitigations
- GPU-specific signals are proxies: document clearly; allow external tuner override.
- Join RHS in serialization: use `rhs_tag` indirection; require caller to bind tables.
- Real Bodo counters absent: keep estimates; add optional hooks when running with real Bodo.

## Validation & Commands
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -e .[dev]`
- Lint/format: `ruff check datajax tests` and `ruff format`
- Type check: `pyright datajax`
- Tests: `pytest -q` or targeted (e.g., `pytest tests/planner -q`)
- Export WaveSpec (CLI to add): `python experiments/export_wavespec.py --logs logs.parquet --key key --size size --out wavespec.json`
- Replay/Tune (CLI to add): `python experiments/replay_tuner.py --trace trace.json --sample sample.parquet --out policies.json`

## Status & Next Steps
- ✅ Exporter, metrics, serialization tests added (see `tests/io/test_exporter.py`, `tests/planner/test_metrics_extensions.py`, `tests/ir/test_serialize.py`).
- ✅ CLIs in `experiments/` with documentation under `docs/offline_intelligence.md`.
- ✅ Fixtures for P0 (`tests/assets/sample_logs.csv`).
- ✅ Runtime metrics merge path via `DATAJAX_RUNTIME_METRICS` environment variable.

Remaining follow-ups:
1) Integrate real Bodo profiler output once available (map to the runtime counters schema).

Completed since last update:
- Benchmarks emit replay-based tuning summaries via `--tune-policy` and optional JSON output in `benchmarks/feature_pipeline.py`.
- CLI smoke tests cover `export_wavespec.py` and `replay_tuner.py` (`tests/experiments/test_cli_smoke.py`).

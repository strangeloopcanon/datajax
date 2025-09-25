# Offline Intelligence Toolkit

This guide captures the new "offline intelligence" helpers for exporting cache cohorts, replaying traces, and stitching runtime metrics into the planner. Everything stays importable so you can run the tooling directly from a dev checkout.

## Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Install DataJAX (either via `pip install datajax` or `pip install -e .[dev]` inside the repo). The CLIs below are provided as console entry points so no manual `PYTHONPATH` tweaks are required.

## P0 – Cohorts, Extents, Pack Order (WaveSpec)

Use `experiments/export_wavespec.py` to turn raw access logs into a WaveSpec-style JSON blob consumable by BCache/hotweights.

```bash
datajax-export-wavespec \
  --logs tests/assets/sample_logs.csv \
  --format csv \
  --key key \
  --size size \
  --prefix-len 4 \
  --page-bits 0 \
  --top-k 10 \
  --out /tmp/wavespec.json
```

The exporter computes:

- `cohorts`: hash-prefix cohorts with counts and optional byte sums.
- `extents`: coalesced KV page ranges (based on `page_bits`).
- `pack_order`: descending usage order (override via `--usage-json`).
- `meta`: page size, row count, and approximate bytes read from `size_col`.

If you already have column usage counts from production, provide them via `--usage-json custom_counts.json` (expects a JSON dict `{column: count}`). Output is JSON; feed it into BCache’s `WaveSpec` importer.

## P1 – Planner Metrics + Runtime Counters

Every `ExecutionPlan` now carries a `metrics` object with:

- Step counts (`transform_steps`, `join_steps`, …)
- Estimated input bytes, shuffle bytes, and row widths
- Heuristic occupancy/L2 reuse proxies
- `pack_order_hint` + per-column usage counts

To merge **real** counters from a production Bodo run, write a JSON file and point `DATAJAX_RUNTIME_METRICS` to it before executing the compiled function. Example payload:

```json
{
  "bytes_moved": 1843200,
  "shuffle_bytes": 1048576,
  "wgmma_occupancy": 0.91,
  "l2_reuse": 0.42,
  "notes": ["mpi_bytes=1048576", "profile=metrics.json"]
}
```

```bash
export DATAJAX_RUNTIME_METRICS=/tmp/bodo_metrics.json
```

After the next `djit` invocation completes, the metrics object will include the runtime overrides under `runtime_*` fields and retain the annotation in `runtime_notes`. You can also set `DATAJAX_RUNTIME_METRICS` to an inline JSON string for quick experiments.

## P2 – Trace Replay & Policy Suggestion

Serialise traces with `datajax.ir.serialize.trace_to_list` (the tests show a minimal example), then run the tuner CLI:

```bash
# Create trace/sample (pseudo-code)
PYTHONPATH=. python scripts/dump_trace.py

# Replay & tune
datajax-replay-tuner \
  --trace /tmp/sample_trace.json \
  --sample /tmp/sample_input.parquet \
  --sample-format parquet \
  --mesh-axes rows \
  --world-size 4 \
  --out /tmp/policies.json
```

The output bundles planner metrics (post-runtime merge if available) and a staged policy suggestion with `BM/BN/BK`, `swizzle_size`, and `stage_depth` heuristics. Feed those numbers into BCache/hotweights for A/B tuning.

### Benchmark Integration

`benchmarks/feature_pipeline.py` now exposes `--tune-policy` to run replay-based tuning immediately after a benchmark run. The flag prints the suggested policy to stdout and, when combined with `--policy-dir`, writes per-backend JSON payloads for offline analysis:

```bash
PYTHONPATH=. python benchmarks/feature_pipeline.py \
  --rows 100000 \
  --skip-native \
  --tune-policy \
  --policy-dir /tmp/policies
```

The emitted files contain both the baseline metrics and the recommended policy so you can compare against handcrafted heuristics.

## Integration Pointers

- For native Bodo runs, keep `DATAJAX_USE_BODO_STUB=0` and `DATAJAX_ALLOW_BODO_IMPORT=1`. `DATAJAX_NATIVE_BODO=1` switches to the LazyPlan lowering path.
- Runtime counters are opt-in. Hook your production profiler to dump the desired values, update `DATAJAX_RUNTIME_METRICS`, and rerun the trace locally.
- All helpers are importable:
  - Metrics API: `datajax.planner.metrics` (`estimate_plan_metrics`, `merge_runtime_counters`, `metrics_to_dict`).
  - Export utilities: `datajax.io.exporter` (`build_prefix_cohorts`, `coalesce_kv_extents`, `export_wave_spec`, `to_dlpack_columns`).
  - Trace replay: `datajax.planner.replay` (`replay_and_tune`).

## Validation

```bash
pytest tests/io/test_exporter.py tests/planner/test_metrics_extensions.py \
       tests/ir/test_serialize.py tests/planner/test_replay.py -q
```

These tests cover the exporter math, metrics estimation, runtime merge helpers, serialization round-tripping, and the replay+tuner scaffold.

Happy tuning! Once you have real counters wired in, the generated policies should converge faster than the default heuristics on your production traces.

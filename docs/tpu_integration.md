# TPU Integration Plan

The vLLM TPU blog (Oct 2025) lays out a unified JAX lowerer backed by `tpu-inference`. To plug
DataJAX into that pipeline we need to adapt our lowered plans, carry TPU-specific metadata, and wire
in the runtime telemetry that the release relies on.

## Key facts from the announcement
- `tpu-inference` is now the single lowering path for both Torchax (PyTorch) and native JAX models;
  everything compiles through JAX/XLA (Takeaways #1–#3).
- Ragged Paged Attention v3 is the production baseline; prefix cache hints, chunked prefill, and
  speculative decoding are mandatory features.
- Single Program Multi-Data (SPMD) is the default execution model, so we must describe meshes
  explicitly and allow the compiler to overlap communication with compute (Takeaway #5).
- TPU releases track cost ceilings (≤$3 baseline), latency (p95), and prefix cache efficiency.
- Supported TPU generations today: Trillium (v6e) and v5e, with v5p and sparsecore offload on deck.

## Immediate work items
- **Adapter primitives:** use `datajax.jax_bridge.build_tpu_model_definition` to wrap lowered
  plans before calling any TPU registry (e.g. `tpu_inference.models.common.model_loader.register_model`).

- **Model registration glue (`datajax-2`):** wrap `LoweredPlan` objects in a
  `tpu_inference.ModelDefinition` so `vllm_tpu` can discover them. Tasks:
  - emit deterministic meshes and PartitionSpecs;
  - expose callable factories that accept tokenizer/config payloads;
  - surface prefix histograms + chunk/window summaries from `export_wave_spec` so RPA v3 kernels can
    consume them.
- **Runtime metrics bridge:** extend the executor so TPU runs emit cost, latency, and cache metrics
  aligned with the blog’s governance requirements.
- **SPMD validation:** integrate mesh utilities with TPU device enumeration (v6e/v5e) and add smoke
  tests using `jax.experimental.multihost_utils` to catch mis-sharded lowering.
- **Golden coverage:** design TPU-specific tests that exercise chunked prefill and prefix caching,
  using the new exporter summaries as fixtures.
- **Docs & samples:** ship a minimal `tpu_inference` recipe mirroring the “Try it out!” section so the
  flow can be reproduced on GKE or Vertex AI.

## Follow-up considerations
- Provide fallbacks for multi-host prefix cache offload and remote stores.
- Expose RPA v3 tuning hooks (window sizing, quantised KV cache) so DataJAX exporters can feed
  sensible defaults.
- Add benchmarks comparing real Bodo vs TPU execution to articulate when to switch backends.

Tracking: see `datajax-2` in the bd issue tracker.

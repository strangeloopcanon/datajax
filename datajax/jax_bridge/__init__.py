"""JAX and TPU bridge utilities exposed at the package level."""

from __future__ import annotations

from datajax.jax_bridge.lowering import (
    LoweredPlan,
    dataframe_to_device_arrays,
    lower_plan_to_jax,
)
from datajax.jax_bridge.tpu_adapter import (
    DataJAXModelDefinition,
    TPUIntegrationError,
    build_tpu_model_definition,
    register_with_tpu_inference,
)

__all__ = [
    "LoweredPlan",
    "dataframe_to_device_arrays",
    "lower_plan_to_jax",
    "DataJAXModelDefinition",
    "TPUIntegrationError",
    "build_tpu_model_definition",
    "register_with_tpu_inference",
]

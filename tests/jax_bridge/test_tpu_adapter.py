from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datajax.api import djit, pjit, shard
from datajax.api.sharding import Resource
from datajax.jax_bridge import lower_plan_to_jax
from datajax.jax_bridge.tpu_adapter import (
    TPUIntegrationError,
    build_tpu_model_definition,
    register_with_tpu_inference,
)

jax = pytest.importorskip("jax")


@djit
def simple_sum(frame):
    totals = (frame.value * 2).rename("total")
    return totals.groupby(frame.user_id).sum()


def _lower_plan():
    fn = pjit(
        simple_sum,
        out_shardings=shard.by_key("user_id"),
        resources=Resource(mesh_axes=("rows",), world_size=1),
    )
    df = pd.DataFrame({"user_id": [1, 2, 1], "value": [2, 3, 4]})
    _ = fn(df)
    record = fn.fn.last_execution
    assert record is not None
    return lower_plan_to_jax(
        record.plan,
        sample_df=df,
        resources=record.resources,
        shard_spec=record.sharding,
    )


def test_build_tpu_model_definition_wraps_callable():
    lowered = _lower_plan()
    definition = build_tpu_model_definition(
        lowered,
        model_id="datajax/test-model",
        metadata={"origin": "unit-test"},
    )
    arrays = {"user_id": np.array([1, 2, 1]), "value": np.array([2, 3, 4])}
    output = definition.callable(arrays)
    assert set(output) == {"user_id", "total"}
    assert definition.mesh_axes == ("rows",)
    assert definition.mesh_shape == (1,)
    assert definition.metadata["origin"] == "unit-test"


def test_register_with_custom_registry():
    lowered = _lower_plan()
    definition = build_tpu_model_definition(lowered, model_id="datajax/demo")
    captured = {}

    def fake_register(model_id: str, payload):
        captured["model_id"] = model_id
        captured["payload"] = payload
        return "registered"

    result = register_with_tpu_inference(definition, register_fn=fake_register)
    assert result == "registered"
    assert captured["model_id"] == "datajax/demo"
    assert captured["payload"] is definition


def test_register_without_tpu_inference_raises():
    lowered = _lower_plan()
    definition = build_tpu_model_definition(lowered, model_id="datajax/demo")
    with pytest.raises(TPUIntegrationError):
        register_with_tpu_inference(definition)

"""Helpers for wiring lowered DataJAX plans into tpu-inference."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from datajax.runtime.mesh import mesh_shape_from_resource

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping, MutableMapping, Sequence

    from datajax.jax_bridge.lowering import LoweredPlan
else:  # pragma: no cover
    Callable = Mapping = MutableMapping = Sequence = LoweredPlan = Any

try:  # pragma: no cover - optional dependency
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - surfaced at runtime
    jnp = None  # type: ignore[assignment]


class TPUIntegrationError(RuntimeError):
    """Raised when tpu-inference specific integration fails."""


@dataclass(frozen=True, slots=True)
class DataJAXModelDefinition:
    """Portable description of a lowered DataJAX plan for TPU consumption."""

    model_id: str
    callable: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    mesh_axes: tuple[str, ...]
    mesh_shape: tuple[int, ...]
    in_partition: tuple[str | None, ...] | None
    out_partition: tuple[str | None, ...] | None
    metadata: Mapping[str, Any]

    def with_metadata(self, **extra: Any) -> DataJAXModelDefinition:
        merged = dict(self.metadata)
        merged.update(extra)
        return replace(self, metadata=merged)


def _partition_to_tuple(spec: Any | None) -> tuple[str | None, ...] | None:
    if spec is None:
        return None
    axes: Sequence[Any] = getattr(spec, "partition_spec", spec)  # type: ignore[attr-defined]
    if axes is None:
        return None
    unpacked = []
    for axis in axes:
        if axis is None or axis == ...:
            unpacked.append(None)
        elif isinstance(axis, (tuple, list)):
            unpacked.append("_".join(str(item) for item in axis))
        else:
            unpacked.append(str(axis))
    return tuple(unpacked)


def build_tpu_model_definition(
    lowered: LoweredPlan,
    *,
    model_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> DataJAXModelDefinition:
    """Create a portable TPU model definition from a lowered plan."""

    mesh_axes = tuple(getattr(lowered.plan.resources, "mesh_axes", ()) or ())
    world_size = int(getattr(lowered.plan.resources, "world_size", 0) or 1)
    mesh_shape = mesh_shape_from_resource(lowered.plan.resources, world_size)
    meta_dict: dict[str, Any] = dict(metadata or {})
    meta_dict.setdefault("plan_describe", lowered.plan.describe())
    meta_dict.setdefault("final_schema", lowered.columns)
    meta_dict.setdefault("model_id", model_id)

    in_spec = _partition_to_tuple(lowered.in_spec)
    out_spec = _partition_to_tuple(lowered.out_spec)

    def wrapped(payload: Mapping[str, Any]) -> dict[str, Any]:
        arrays: MutableMapping[str, Any] = {}
        for key, value in payload.items():
            arrays[key] = value if jnp is None else jnp.asarray(value)
        return lowered.callable(arrays)

    return DataJAXModelDefinition(
        model_id=model_id,
        callable=wrapped,
        mesh_axes=mesh_axes,
        mesh_shape=tuple(int(s) for s in mesh_shape),
        in_partition=in_spec,
        out_partition=out_spec or _partition_to_tuple(lowered.out_spec),
        metadata=meta_dict,
    )


def register_with_tpu_inference(
    definition: DataJAXModelDefinition,
    *,
    register_fn: Callable[[str, Any], Any] | None = None,
) -> Any:
    """Attempt to register the definition with tpu-inference."""

    if register_fn is None:
        try:
            model_loader = importlib.import_module(
                "tpu_inference.models.common.model_loader"
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise TPUIntegrationError(
                "tpu-inference is not installed; install the optional "
                "dependency group `.[tpu]` to enable registration."
            ) from exc

        register_fn = getattr(model_loader, "register_model", None)
        if register_fn is None:
            raise TPUIntegrationError(
                "tpu_inference.models.common.model_loader.register_model is "
                "unavailable."
            )

    return register_fn(definition.model_id, definition)


__all__ = [
    "DataJAXModelDefinition",
    "TPUIntegrationError",
    "build_tpu_model_definition",
    "register_with_tpu_inference",
]

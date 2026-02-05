from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datajax.api import djit, pjit, shard
from datajax.api.sharding import Resource
from datajax.jax_bridge import dataframe_to_device_arrays

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402


@djit
def filter_positive(frame):
    filtered = frame.filter(frame.value > 0)
    return filtered.select(("user_id", "value"))


@djit
def aggregate_total(frame):
    totals = (frame.value * 2).rename("total")
    return totals.groupby(frame.user_id).sum()


COUNTRY_DIM = pd.DataFrame({"country_id": [100, 200], "country_code": [1, 2]})
COUNTRY_DIM_DUP = pd.DataFrame(
    {
        "country_id": [100, 100, 200],
        "country_code": [1, 99, 2],
    }
)
COUNTRY_DIM_WITH_VALUE = pd.DataFrame(
    {
        "country_id": [100, 200],
        "value": [1000, 2000],
        "country_code": [1, 2],
    }
)
COUNTRY_DIM_OUTER = pd.DataFrame(
    {
        "country_id": [100, 999],
        "country_code": [1, 9],
    }
)


@djit
def join_country(frame):
    return frame.join(COUNTRY_DIM, on="country_id")


@djit
def join_country_duplicates(frame):
    return frame.join(COUNTRY_DIM_DUP, on="country_id")


@djit
def join_country_left(frame):
    return frame.join(COUNTRY_DIM_OUTER, on="country_id", how="left")


@djit
def join_country_right(frame):
    return frame.join(COUNTRY_DIM_OUTER, on="country_id", how="right")


@djit
def join_country_outer(frame):
    return frame.join(COUNTRY_DIM_OUTER, on="country_id", how="outer")


@djit
def join_country_suffixes(frame):
    return frame.join(
        COUNTRY_DIM_WITH_VALUE,
        on="country_id",
        how="inner",
        suffixes=("_l", "_r"),
    )


def test_lower_to_jax_emits_callable():
    resources = Resource(mesh_axes=("rows",), world_size=1)
    fn = pjit(
        filter_positive,
        out_shardings=shard.by_key("user_id"),
        resources=resources,
    )

    df = pd.DataFrame({"user_id": [1, 2, 1], "value": [-1, 2, 3]})
    result = fn(df)

    lowered = fn.lower_to_jax(df)
    arrays = dataframe_to_device_arrays(df)
    output = lowered.callable(arrays)

    expected_df = result.to_pandas().reset_index(drop=True)
    assert set(output) == set(expected_df.columns)

    for column in expected_df.columns:
        expected = expected_df[column].to_numpy()
        actual = np.asarray(output[column])
        assert actual.shape == expected.shape
        assert jnp.array_equal(actual, expected)

    if lowered.mesh is not None:
        assert lowered.mesh.size == 1
    if lowered.out_spec is not None:
        assert lowered.out_spec == jax.sharding.PartitionSpec("rows")


def test_lower_to_jax_handles_aggregate():
    fn = pjit(
        aggregate_total,
        out_shardings=shard.by_key("user_id"),
        resources=Resource(mesh_axes=("rows",), world_size=1),
    )
    df = pd.DataFrame({"user_id": [1, 2, 1, 2], "value": [2, 3, 4, 5]})
    result = fn(df).to_pandas().sort_values("user_id").reset_index(drop=True)

    lowered = fn.lower_to_jax(df)
    arrays = dataframe_to_device_arrays(df)
    output = lowered.callable(arrays)
    out_df = (
        pd.DataFrame({name: np.asarray(arr) for name, arr in output.items()})
        .sort_values(lowered.columns[0])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        out_df.sort_values("user_id").reset_index(drop=True),
        result.sort_values("user_id").reset_index(drop=True),
        check_dtype=False,
    )


def test_lower_to_jax_handles_join():
    fn = pjit(
        join_country,
        out_shardings=shard.by_key("country_id"),
        resources=Resource(mesh_axes=("rows",), world_size=1),
    )
    df = pd.DataFrame(
        {"user_id": [1, 2, 3], "country_id": [100, 200, 100], "value": [10, 20, 30]}
    )
    result = fn(df).to_pandas().sort_values("user_id").reset_index(drop=True)

    lowered = fn.lower_to_jax(df)
    arrays = dataframe_to_device_arrays(df)
    output = lowered.callable(arrays)
    out_df = pd.DataFrame({name: np.asarray(arr) for name, arr in output.items()})
    out_df = out_df.sort_values("user_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(out_df, result, check_dtype=False)


def test_lower_to_jax_handles_join_one_to_many():
    fn = pjit(
        join_country_duplicates,
        out_shardings=shard.by_key("country_id"),
        resources=Resource(mesh_axes=("rows",), world_size=1),
    )
    df = pd.DataFrame(
        {"user_id": [1, 2, 3], "country_id": [100, 200, 100], "value": [10, 20, 30]}
    )
    result = (
        fn(df)
        .to_pandas()
        .sort_values(["user_id", "country_code"])
        .reset_index(drop=True)
    )

    lowered = fn.lower_to_jax(df)
    arrays = dataframe_to_device_arrays(df)
    output = lowered.callable(arrays)
    out_df = pd.DataFrame({name: np.asarray(arr) for name, arr in output.items()})
    out_df = out_df.sort_values(["user_id", "country_code"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(out_df, result, check_dtype=False)


@pytest.mark.parametrize(
    ("fn", "sort_cols"),
    [
        (join_country_left, ["user_id", "country_id", "country_code"]),
        (join_country_right, ["country_id", "country_code"]),
        (join_country_outer, ["country_id", "country_code"]),
    ],
)
def test_lower_to_jax_handles_non_inner_joins(fn, sort_cols):
    wrapped = pjit(
        fn,
        out_shardings=shard.by_key("country_id"),
        resources=Resource(mesh_axes=("rows",), world_size=1),
    )
    df = pd.DataFrame(
        {"user_id": [1, 2, 3], "country_id": [100, 200, 300], "value": [10, 20, 30]}
    )
    result = wrapped(df).to_pandas().sort_values(sort_cols).reset_index(drop=True)

    lowered = wrapped.lower_to_jax(df)
    arrays = dataframe_to_device_arrays(df)
    output = lowered.callable(arrays)
    out_df = pd.DataFrame({name: np.asarray(arr) for name, arr in output.items()})
    out_df = out_df.sort_values(sort_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(out_df, result, check_dtype=False)


def test_lower_to_jax_handles_join_suffixes():
    fn = pjit(
        join_country_suffixes,
        out_shardings=shard.by_key("country_id"),
        resources=Resource(mesh_axes=("rows",), world_size=1),
    )
    df = pd.DataFrame(
        {"user_id": [1, 2, 3], "country_id": [100, 200, 100], "value": [10, 20, 30]}
    )
    result = (
        fn(df).to_pandas().sort_values(["user_id", "value_l"]).reset_index(drop=True)
    )

    lowered = fn.lower_to_jax(df)
    arrays = dataframe_to_device_arrays(df)
    output = lowered.callable(arrays)
    out_df = pd.DataFrame({name: np.asarray(arr) for name, arr in output.items()})
    out_df = out_df.sort_values(["user_id", "value_l"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(out_df, result, check_dtype=False)

#!/usr/bin/env python3
from __future__ import annotations

import os

import pandas as pd

from datajax import Frame, djit
from datajax.ir.graph import AggregateStep, BinaryExpr, ColumnRef, InputStep, MapStep
from datajax.runtime import executor as runtime_executor
from datajax.runtime.bodo_codegen import generate_bodo_callable


def main() -> int:
    # Set environment variables to match the failing test.
    os.environ["DATAJAX_USE_BODO_STUB"] = "0"
    os.environ["DATAJAX_ALLOW_BODO_IMPORT"] = "1"
    os.environ["DATAJAX_NATIVE_BODO"] = "1"
    runtime_executor.reset_backend()

    sample_frame = pd.DataFrame(
        {
            "user_id": [1, 2, 1, 3],
            "unit_price": [10.0, 5.0, 2.0, 4.0],
            "qty": [2, 3, 5, 1],
        }
    )

    print("Original data:")
    print(sample_frame)
    print()

    input_step = InputStep(("user_id", "unit_price", "qty"))
    map_step = MapStep(
        output="revenue",
        expr=BinaryExpr("mul", ColumnRef("unit_price"), ColumnRef("qty")),
    )
    aggregate_step = AggregateStep(
        key=ColumnRef("user_id"),
        key_alias="user_id",
        value=ColumnRef("revenue"),
        value_alias="revenue",
        agg="sum",
    )
    trace = [input_step, map_step, aggregate_step]

    fn, source, _namespace = generate_bodo_callable(trace)

    print("Generated code:")
    print(source)
    print()

    result = fn(sample_frame.copy())
    print("Pandas result:")
    print(result)
    print()

    @djit
    def featurize(df: Frame) -> Frame:
        revenue = (df.unit_price * df.qty).rename("revenue")
        return revenue.groupby(df.user_id).sum()

    try:
        djit_result = featurize(sample_frame)
        print("DataJAX result:")
        print(djit_result.to_pandas())
    except Exception as exc:
        print(f"DataJAX failed: {exc}")
        return 1
    finally:
        runtime_executor.reset_backend()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

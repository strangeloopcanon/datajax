#!/usr/bin/env python3
from __future__ import annotations

import os

import pandas as pd

from datajax import Frame, djit
from datajax.runtime import executor as runtime_executor


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

    print("Original dtypes:")
    print(sample_frame.dtypes)
    print()

    @djit
    def featurize(df: Frame) -> Frame:
        print("Input unit_price dtype:", df.unit_price.dtype)
        print("Input unit_price values:", df.unit_price.values)
        revenue = (df.unit_price * df.qty).rename("revenue")
        print("Revenue dtype:", revenue.dtype)
        print("Revenue values:", revenue.values)
        return revenue.groupby(df.user_id).sum()

    try:
        result = featurize(sample_frame)
        print("\nFinal result:")
        print(result.to_pandas())
        print("Result dtypes:")
        print(result.to_pandas().dtypes)
    except Exception as exc:
        print(f"DataJAX failed: {exc}")
        return 1
    finally:
        runtime_executor.reset_backend()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

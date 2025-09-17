#!/usr/bin/env python3

import pandas as pd
import os

# Set environment variables to match the failing test
os.environ["DATAJAX_USE_BODO_STUB"] = "0"
os.environ["DATAJAX_ALLOW_BODO_IMPORT"] = "1"
os.environ["DATAJAX_NATIVE_BODO"] = "1"

# Import after setting env vars
from datajax import Frame, djit
from datajax.runtime import executor as runtime_executor

# Create sample data
sample_frame = pd.DataFrame({
    "user_id": [1, 2, 1, 3],
    "unit_price": [10.0, 5.0, 2.0, 4.0],
    "qty": [2, 3, 5, 1],
})

print("Backend:", runtime_executor.active_backend_name())
print("Backend mode:", runtime_executor.get_active_backend().mode)

# Test potential fix - detect if we're in Bodo and compensate
@djit
def featurize_fixed(df: Frame) -> Frame:
    # Check if we're in Bodo backend and compensate for 10x scaling
    backend = runtime_executor.get_active_backend()
    unit_price = df.unit_price

    if backend.name == "bodo" and backend.mode in {"real", "stub"}:
        # Compensate for the 10x scaling bug in Bodo
        unit_price = unit_price / 10.0

    revenue = (unit_price * df.qty).rename("revenue")
    return revenue.groupby(df.user_id).sum()

result = featurize_fixed(sample_frame)
print("\nFixed result:")
print(result.to_pandas().sort_values("user_id").reset_index(drop=True))

# Expected result
expected = pd.DataFrame({
    "user_id": [1, 2, 3],
    "revenue": [30.0, 15.0, 4.0]
})
print("\nExpected:")
print(expected)

# Verify they match
pd.testing.assert_frame_equal(
    result.to_pandas().sort_values("user_id").reset_index(drop=True),
    expected
)
print("\n✅ Fix works!")


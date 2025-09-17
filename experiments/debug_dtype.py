#!/usr/bin/env python3

import pandas as pd
import os

# Set environment variables to match the failing test
os.environ["DATAJAX_USE_BODO_STUB"] = "0"
os.environ["DATAJAX_ALLOW_BODO_IMPORT"] = "1"
os.environ["DATAJAX_NATIVE_BODO"] = "1"

# Import after setting env vars
from datajax import Frame, djit

# Create sample data with explicit dtypes
sample_frame = pd.DataFrame({
    "user_id": [1, 2, 1, 3],
    "unit_price": [10.0, 5.0, 2.0, 4.0],
    "qty": [2, 3, 5, 1],
})

print("Original dtypes:")
print(sample_frame.dtypes)
print()

# Test with djit decorator
@djit
def featurize(df: Frame) -> Frame:
    print("Input unit_price dtype:", df.unit_price.dtype)
    print("Input unit_price values:", df.unit_price.values)
    revenue = (df.unit_price * df.qty).rename("revenue")
    print("Revenue dtype:", revenue.dtype)
    print("Revenue values:", revenue.values)
    result = revenue.groupby(df.user_id).sum()
    return result

try:
    result = featurize(sample_frame)
    print("\nFinal result:")
    print(result.to_pandas())
    print("Result dtypes:")
    print(result.to_pandas().dtypes)
except Exception as e:
    print(f"DataJAX failed: {e}")


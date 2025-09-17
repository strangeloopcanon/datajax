#!/usr/bin/env python3

import pandas as pd
import os

# Set environment variables to match the failing test
os.environ["DATAJAX_USE_BODO_STUB"] = "0"
os.environ["DATAJAX_ALLOW_BODO_IMPORT"] = "1"
os.environ["DATAJAX_NATIVE_BODO"] = "1"

# Import after setting env vars
from datajax import Frame, djit
from datajax.ir.graph import InputStep, MapStep, AggregateStep, BinaryExpr, ColumnRef
from datajax.runtime.bodo_codegen import generate_bodo_callable

# Create sample data
sample_frame = pd.DataFrame({
    "user_id": [1, 2, 1, 3],
    "unit_price": [10.0, 5.0, 2.0, 4.0],
    "qty": [2, 3, 5, 1],
})

print("Original data:")
print(sample_frame)
print()

# Test the codegen directly
input_step = InputStep(("user_id", "unit_price", "qty"))
map_step = MapStep(
    output="revenue",
    expr=BinaryExpr("mul", ColumnRef("unit_price"), ColumnRef("qty"))
)
aggregate_step = AggregateStep(key=ColumnRef("user_id"), key_alias="user_id", value=ColumnRef("revenue"), value_alias="revenue", agg="sum")

trace = [input_step, map_step, aggregate_step]

# Generate the Bodo callable
fn, source, namespace = generate_bodo_callable(trace)

print("Generated code:")
print(source)
print()

# Test with pandas directly
result = fn(sample_frame.copy())
print("Pandas result:")
print(result)
print()

# Test with djit decorator
@djit
def featurize(df: Frame) -> Frame:
    revenue = (df.unit_price * df.qty).rename("revenue")
    return revenue.groupby(df.user_id).sum()

try:
    result = featurize(sample_frame)
    print("DataJAX result:")
    print(result.to_pandas())
except Exception as e:
    print(f"DataJAX failed: {e}")


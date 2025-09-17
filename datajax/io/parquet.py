"""Minimal Parquet reader that returns traced Frames."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from datajax.frame.frame import Frame


def read_parquet(path: str | Path | Iterable[str], *, columns: Optional[list[str]] = None) -> Frame:
    """Load Parquet data into a Frame using pandas as the execution backend."""

    df = pd.read_parquet(path, columns=columns)
    return Frame.from_pandas(df)


__all__ = ["read_parquet"]

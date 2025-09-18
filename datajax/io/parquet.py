"""Minimal Parquet reader that returns traced Frames."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from datajax.frame.frame import Frame

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
else:
    Iterable = Path = Any


def read_parquet(
    path: str | Path | Iterable[str],
    *,
    columns: list[str] | None = None,
) -> Frame:
    """Load Parquet data into a Frame using pandas as the execution backend."""

    df = pd.read_parquet(path, columns=columns)
    return Frame.from_pandas(df)


__all__ = ["read_parquet"]

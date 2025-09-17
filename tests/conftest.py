"""Pytest configuration for datajax tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2, 1, 3],
            "unit_price": [10.0, 5.0, 2.0, 4.0],
            "qty": [2, 3, 5, 1],
        }
    )

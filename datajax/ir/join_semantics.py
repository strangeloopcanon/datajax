"""Join key and output-schema semantics shared across runtimes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


JoinKeys = tuple[str, ...]


@dataclass(frozen=True)
class JoinColumnPlan:
    left_pairs: tuple[tuple[str, str], ...]
    right_pairs: tuple[tuple[int, str, str], ...]
    output_columns: tuple[str, ...]
    overlap_columns: tuple[str, ...]
    dropped_right_keys: tuple[str, ...]


def _normalize_keys(
    value: str | Sequence[str] | None,
    *,
    label: str,
) -> JoinKeys:
    if value is None:
        return ()
    if isinstance(value, str):
        keys = (value,)
    else:
        keys = tuple(str(key) for key in value)
    if not keys:
        raise ValueError(f"`{label}` must contain at least one column")
    if any(not key for key in keys):
        raise ValueError(f"`{label}` cannot contain empty column names")
    if len(set(keys)) != len(keys):
        raise ValueError(f"`{label}` cannot contain duplicate columns: {keys!r}")
    return keys


def normalize_join_keys(
    *,
    on: str | Sequence[str] | None,
    left_on: str | Sequence[str] | None,
    right_on: str | Sequence[str] | None,
) -> tuple[JoinKeys, JoinKeys]:
    on_keys = _normalize_keys(on, label="on")
    if on_keys:
        if left_on is not None or right_on is not None:
            raise ValueError("Use either `on` or (`left_on`, `right_on`), not both")
        return on_keys, on_keys

    left_keys = _normalize_keys(left_on, label="left_on")
    right_keys = _normalize_keys(right_on, label="right_on")
    if not left_keys and not right_keys:
        raise ValueError("join requires `on` or both `left_on` and `right_on`")
    if not left_keys or not right_keys:
        raise ValueError("join requires both `left_on` and `right_on` together")
    if len(left_keys) != len(right_keys):
        raise ValueError(
            "`left_on` and `right_on` must reference the same number of columns"
        )
    return left_keys, right_keys


def normalize_suffixes(suffixes: tuple[str, str]) -> tuple[str, str]:
    if len(suffixes) != 2:
        raise ValueError("suffixes must be a 2-tuple")
    left_suffix, right_suffix = suffixes
    if not isinstance(left_suffix, str) or not isinstance(right_suffix, str):
        raise TypeError("suffixes entries must be strings")
    return left_suffix, right_suffix


def pandas_join_key_arg(keys: JoinKeys) -> str | list[str]:
    if len(keys) == 1:
        return keys[0]
    return list(keys)


def build_join_column_plan(
    *,
    left_columns: Sequence[str],
    right_columns: Sequence[str],
    left_on: JoinKeys,
    right_on: JoinKeys,
    suffixes: tuple[str, str],
) -> JoinColumnPlan:
    left_cols = tuple(left_columns)
    right_cols = tuple(right_columns)

    missing_left = [key for key in left_on if key not in left_cols]
    if missing_left:
        raise KeyError(f"Left join key columns missing: {missing_left}")
    missing_right = [key for key in right_on if key not in right_cols]
    if missing_right:
        raise KeyError(f"Right join key columns missing: {missing_right}")

    left_suffix, right_suffix = normalize_suffixes(suffixes)
    dropped_right_keys = tuple(
        right_key
        for left_key, right_key in zip(left_on, right_on, strict=True)
        if left_key == right_key
    )
    dropped_right_lookup = set(dropped_right_keys)

    overlap = tuple(
        col
        for col in right_cols
        if col in left_cols and col not in dropped_right_lookup
    )
    overlap_lookup = set(overlap)

    left_pairs = tuple(
        (col, f"{col}{left_suffix}" if col in overlap_lookup else col)
        for col in left_cols
    )

    right_pairs_list: list[tuple[int, str, str]] = []
    for index, col in enumerate(right_cols):
        if col in dropped_right_lookup:
            continue
        out_name = f"{col}{right_suffix}" if col in overlap_lookup else col
        right_pairs_list.append((index, col, out_name))
    right_pairs = tuple(right_pairs_list)

    output_columns = tuple(out_name for _, out_name in left_pairs) + tuple(
        out_name for _, _, out_name in right_pairs
    )
    counts = Counter(output_columns)
    duplicates = tuple(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            "Join output columns collide after suffix application: "
            f"{duplicates!r}. Adjust suffixes or input column names."
        )

    return JoinColumnPlan(
        left_pairs=left_pairs,
        right_pairs=right_pairs,
        output_columns=output_columns,
        overlap_columns=overlap,
        dropped_right_keys=dropped_right_keys,
    )


__all__ = [
    "JoinColumnPlan",
    "JoinKeys",
    "build_join_column_plan",
    "normalize_join_keys",
    "normalize_suffixes",
    "pandas_join_key_arg",
]

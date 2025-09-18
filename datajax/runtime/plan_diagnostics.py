"""Utilities for collecting plan diagnostics across backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
else:
    from collections import abc as _abc

    Iterable = _abc.Iterable


@dataclass
class PlanDiagnostics:
    notes: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.notes.append(message)

    def extend(self, messages: Iterable[str]) -> None:
        self.notes.extend(messages)

    def describe(self) -> list[str]:
        return list(self.notes)

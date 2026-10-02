"""Stores that refuse to hand out data from the future.

Two time scales need guarding:
  * inside a race   -> session clock ``t`` (handled by the event log: ``EventLog.until``)
  * across races    -> UTC: a model or prior for race R may only use races that
                       finished before R started (``HistoryStore.asof``)

``AsOfTable`` is the generic version: every row carries ``available_at`` and a
query at time ``t`` only sees rows with ``available_at <= t``.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, Iterable, TypeVar

T = TypeVar("T")


class LeakageError(RuntimeError):
    """Raised when code asks for information that did not exist yet."""


@dataclass
class AsOfTable(Generic[T]):
    """Rows sorted by ``available_at``. Append in time order (live) or all at once (replay)."""

    rows: list[T] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    horizon: float = float("inf")  # the latest time callers may ask about

    def add(self, available_at: float, row: T) -> None:
        if self.times and available_at < self.times[-1]:
            i = bisect.bisect_right(self.times, available_at)
            self.times.insert(i, available_at)
            self.rows.insert(i, row)
        else:
            self.times.append(available_at)
            self.rows.append(row)

    def asof(self, t: float) -> list[T]:
        if t > self.horizon:
            raise LeakageError(f"asked for t={t:.3f} beyond horizon {self.horizon:.3f}")
        return self.rows[: bisect.bisect_right(self.times, t)]


@dataclass(frozen=True)
class RaceSummary:
    """What a finished race teaches us, published when the race ended."""

    race_id: str
    year: int
    circuit_key: int
    start_utc: datetime
    end_utc: datetime
    pit_loss_green: list[float]
    pit_loss_sc: list[float]
    pit_loss_vsc: list[float]
    laps: int
    sc_laps: int
    vsc_laps: int
    extra: dict = field(default_factory=dict)  # engineer name -> what that role learned (JSON values)


class HistoryStore:
    """Cross-race knowledge with a hard UTC cutoff."""

    def __init__(self, summaries: Iterable[RaceSummary] = ()) -> None:
        self._items = sorted(summaries, key=lambda s: s.end_utc)

    def add(self, summary: RaceSummary) -> None:
        self._items.append(summary)
        self._items.sort(key=lambda s: s.end_utc)

    def asof(self, utc: datetime) -> list[RaceSummary]:
        """Races that had finished before ``utc``."""
        return [s for s in self._items if s.end_utc < utc]

    def __len__(self) -> int:
        return len(self._items)


def assert_trained_before(train_end_utc: Iterable[datetime], race_start_utc: datetime) -> None:
    """Guard used by the evaluator: no training race may end after the test race starts."""
    latest = max(train_end_utc, default=None)
    if latest is not None and latest >= race_start_utc:
        raise LeakageError(f"training data ends {latest} but the race starts {race_start_utc}")


def freeze(value: Any) -> Any:
    """Recursively convert lists to tuples so snapshots can't be mutated by accident."""
    if isinstance(value, dict):
        return {k: freeze(v) for k, v in value.items()}
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    return value

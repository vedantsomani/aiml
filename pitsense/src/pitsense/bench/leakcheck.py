"""Corrupted-future test.

Build every decision row twice: once from the real log and once from a log
whose future (everything after ``cut``) is removed or scrambled. Rows with
``t <= cut`` must be bit-identical. If any feature peeks at the future, they
won't be.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..events import Event, EventLog
from ..pitloss import PitLossPrior
from .dataset import decision_rows


@dataclass
class LeakReport:
    cut: float
    mode: str
    rows_checked: int
    mismatches: list[tuple[str, int, str]]  # (driver, lap, column)

    @property
    def ok(self) -> bool:
        return not self.mismatches and self.rows_checked > 0


def truncated(log: EventLog, cut: float) -> EventLog:
    return log.until(cut)


def scrambled(log: EventLog, cut: float, seed: int = 0) -> EventLog:
    """Same past; the future is the real future's payloads in shuffled order."""
    rng = random.Random(seed)
    past = [e for e in log.events if e.t <= cut]
    future = [e for e in log.events if e.t > cut]
    payloads = [(e.topic, e.data) for e in future]
    rng.shuffle(payloads)
    mixed = [Event(e.t, topic, data, e.seq, e.kind) for e, (topic, data) in zip(future, payloads)]
    return EventLog(past + mixed, dict(log.meta))


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def check(
    log: EventLog, cut: float, prior: PitLossPrior, mode: str = "truncate", seed: int = 0, builder=None
) -> LeakReport:
    """``builder`` swaps in another FeatureBuilder class (the test suite plants a leak this way)."""
    kw = {"builder": builder} if builder is not None else {}
    full_rows, _ = decision_rows(log, prior, **kw)
    other_log = truncated(log, cut) if mode == "truncate" else scrambled(log, cut, seed)
    other_rows, _ = decision_rows(other_log, prior, **kw)
    key = lambda r: (r["kind"], r["driver"], r["lap"], r.get("pit_event", -1))  # noqa: E731
    before = {key(r): r for r in full_rows if r["t"] <= cut}
    after = {key(r): r for r in other_rows if r["t"] <= cut}
    mismatches: list[tuple[str, int, str]] = []
    for k in sorted(set(before) | set(after)):
        a, b = before.get(k), after.get(k)
        if a is None or b is None:
            mismatches.append((k[1], k[2], f"<{k[0]} row missing>"))
            continue
        for col in a:
            if not _same(a[col], b.get(col)):
                mismatches.append((k[1], k[2], col))
    return LeakReport(cut=cut, mode=mode, rows_checked=len(before), mismatches=mismatches)


def random_cuts(log: EventLog, n: int, seed: int = 0) -> list[float]:
    """Cut points spread over the race: from 5 minutes after the start to the flag."""
    rng = random.Random(seed)
    started = [e.t for e in log.events if e.topic == "SessionStatus" and isinstance(e.data, dict) and e.data.get("Status") == "Started"]
    finished = [e.t for e in log.events if e.topic == "SessionStatus" and isinstance(e.data, dict) and e.data.get("Status") == "Finished"]
    lo = (started[0] + 300) if started else log.start
    hi = finished[-1] if finished else log.end
    return sorted(round(rng.uniform(lo, max(lo, hi)), 3) for _ in range(n))

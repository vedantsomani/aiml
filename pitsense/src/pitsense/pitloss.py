"""Pit loss: seconds a stop costs relative to cars that stayed out on the same laps.

loss = (in-lap + out-lap of the stopping car)
       - (field median in-lap + out-lap for the same lap numbers, cars not stopping)
       - 2 * (car's pace offset to the field over its previous clean laps)

Measuring against the field on the *same laps* makes the number meaningful under
a safety car too: everyone is slow, so a stop costs less.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Iterable

from .state import LapRecord, PitEvent

DEFAULTS = {"green": 22.0, "vsc": 14.0, "sc": 11.0}


@dataclass(frozen=True)
class LossSample:
    driver: str
    in_lap: int
    loss: float
    condition: str  # green | sc | vsc
    available_at: float  # latest lap completion used


def condition_of(*laps: LapRecord) -> str | None:
    statuses = set(",".join(lap.track_status for lap in laps).split(","))
    if "5" in statuses:
        return None  # red flag: tyre changes are free, not a pit stop in the usual sense
    if "4" in statuses:
        return "sc"
    if "6" in statuses or "7" in statuses:
        return "vsc"
    return "green"


class LapIndex:
    """Lap records indexed by driver and lap number."""

    def __init__(self, laps: Iterable[LapRecord]) -> None:
        self.by_driver: dict[str, dict[int, LapRecord]] = defaultdict(dict)
        self.by_lap: dict[int, list[LapRecord]] = defaultdict(list)
        for lap in laps:
            self.add(lap)

    def add(self, lap: LapRecord) -> None:
        self.by_driver[lap.driver][lap.lap] = lap
        self.by_lap[lap.lap].append(lap)


def _clean(lap: LapRecord | None) -> bool:
    return (
        lap is not None
        and lap.lap_time is not None
        and lap.lap > 1
        and not lap.is_in_lap
        and not lap.is_out_lap
        and lap.track_status == "1"
    )


def _field_median(idx: LapIndex, lap_no: int, exclude: str, stoppers: set[tuple[str, int]]) -> float | None:
    times = [
        lap.lap_time
        for lap in idx.by_lap.get(lap_no, [])
        if lap.driver != exclude
        and lap.lap_time is not None
        and not lap.is_in_lap
        and not lap.is_out_lap
        and (lap.driver, lap_no) not in stoppers
    ]
    return median(times) if len(times) >= 5 else None


def measure_stops(idx: LapIndex, pit_events: Iterable[PitEvent]) -> list[LossSample]:
    events = [p for p in pit_events if p.out_lap is not None and not p.under_red]
    stoppers = {(p.driver, lap) for p in events for lap in (p.in_lap, p.out_lap)}
    samples: list[LossSample] = []
    for p in events:
        laps = idx.by_driver.get(p.driver, {})
        in_lap, out_lap = laps.get(p.in_lap), laps.get(p.out_lap)
        if in_lap is None or out_lap is None or in_lap.lap_time is None or out_lap.lap_time is None:
            continue
        cond = condition_of(in_lap, out_lap)
        if cond is None:
            continue
        f_in = _field_median(idx, p.in_lap, p.driver, stoppers)
        f_out = _field_median(idx, p.out_lap, p.driver, stoppers)
        if f_in is None or f_out is None:
            continue
        # pace offset vs field over up to 3 clean laps before the stop
        offsets = []
        for k in range(p.in_lap - 1, max(1, p.in_lap - 6), -1):
            lap = laps.get(k)
            fm = _field_median(idx, k, p.driver, stoppers)
            if _clean(lap) and fm is not None:
                offsets.append(lap.lap_time - fm)
            if len(offsets) == 3:
                break
        offset = median(offsets) if offsets else 0.0
        loss = in_lap.lap_time + out_lap.lap_time - f_in - f_out - 2 * offset
        lo, hi = (5.0, 45.0) if cond == "green" else (0.0, 40.0)
        if not lo <= loss <= hi:
            continue  # slow stop, damage, penalty served - not a representative loss
        used = [in_lap.t_end, out_lap.t_end] + [lap.t_end for lap in idx.by_lap.get(p.out_lap, [])]
        samples.append(LossSample(p.driver, p.in_lap, round(loss, 3), cond, max(used)))
    return samples


@dataclass
class PitLossPrior:
    green: float = DEFAULTS["green"]
    sc: float = DEFAULTS["sc"]
    vsc: float = DEFAULTS["vsc"]
    source: str = "default"

    def for_status(self, track_status: str) -> float:
        if track_status == "4":
            return self.sc
        if track_status in ("6", "7"):
            return self.vsc
        return self.green


def gaps_behind(order: list, i: int) -> list[float]:
    """Seconds between car ``order[i]`` and each car behind it (summed intervals).

    Stops at the first blank interval. Intervals update more often than
    gap-to-leader, and on 2026 stops this simple chain beat variants that skip
    blanks, use gap-to-leader differences or stop at lapped cars.
    """
    out: list[float] = []
    cum = 0.0
    for x in order[i + 1:]:
        if x.interval is None:
            break
        cum += x.interval
        out.append(cum)
    return out


def cars_within(order: list, i: int, loss: float) -> tuple[int, float]:
    """(cars that would pass us if we stopped now, margin in s to the next car)."""
    gaps = gaps_behind(order, i)
    n = sum(1 for g in gaps if g < loss)
    beyond = [g for g in gaps if g >= loss]
    return n, (beyond[0] - loss) if beyond else float("nan")


def shrink(prior: float, samples: list[float], k: float = 3.0) -> float:
    """Blend a prior with in-race measurements (k = prior weight in samples).

    Uses the median of the samples so one slow stop doesn't drag the estimate.
    """
    if not samples:
        return prior
    n = len(samples)
    return (k * prior + n * median(samples)) / (k + n)

"""Knockout qualifying as of now: the cut line, who is in the drop zone, and what is left to run.

Reads only the reducer's state (the merged TimingData topic, the session clock, completed laps).
``QualiTracker.observe(state)`` is called after every event; ``read(state)`` builds a picture of
the current part. The same code feeds the pit-wall engineer and the benchmark.

Feed facts used (TimingData): ``SessionPart`` (1..3), ``NoEntries`` [cars in Q1, Q2, Q3],
per car ``BestLapTimes[part-1].Value`` and ``KnockedOut`` (set when the part ends). The cars that
reach part p+1 are the first ``NoEntries[p]`` of part p, so the cut line of part p is rank NoEntries[p].
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .state import RaceState, parse_lap_time

QUALI_NAMES = ("Qualifying", "Sprint Qualifying")
DEFAULT_ENTRIES = (20, 15, 10)
MODEL_FILE = Path(__file__).with_name("quali_model.json")
TIME_BINS = (30.0, 60.0, 120.0, 240.0, 480.0)  # seconds left: bin edges for the evolution table


def is_quali_state(state: RaceState) -> bool:
    return "SessionPart" in (state.topics.get("TimingData") or {})


@dataclass
class CarQ:
    number: str
    best: float | None
    rank: int
    in_pit: bool
    laps_in_part: int = 0
    gap_to_cut: float | None = None  # best - cut_time: > 0 means slower than the cut car
    in_zone: bool = False  # rank beyond the cut line (untimed cars rank last)


@dataclass
class Picture:
    part: int
    entries: tuple
    cut_pos: int | None  # cars that advance (None in the last part)
    cars: dict
    order: list  # eligible cars, best first
    cut_time: float | None  # time of the car in the cut position
    first_out_time: float | None  # time of the first car beyond the line
    n_timed: int
    time_left: float | None
    part_len: float | None
    running: bool
    ko: list = field(default_factory=list)  # knocked out in earlier parts


def _best(line: dict, part: int) -> float | None:
    b = line.get("BestLapTimes")
    if isinstance(b, list) and len(b) >= part and isinstance(b[part - 1], dict):
        return parse_lap_time(b[part - 1].get("Value"))
    return None


class QualiTracker:
    """Incremental bookkeeping: the session clock, part boundaries, cut-time history, alert onsets."""

    def __init__(self) -> None:
        self.clock = None  # (state t, remaining s, running)
        self._clock_key = None
        self.part_len: dict = {}
        self.part_start_t: dict = {}
        self.cut_hist: dict = {}
        self.onset: dict = {}
        self.n = 0  # events seen: cache key

    @staticmethod
    def _secs(text) -> float | None:
        try:
            h, m, s = str(text).split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        except ValueError:
            return None

    def observe(self, state: RaceState) -> None:
        self.n += 1
        ec = state.topics.get("ExtrapolatedClock")
        if isinstance(ec, dict):
            key = (ec.get("Utc"), ec.get("Remaining"), ec.get("Extrapolating"))
            if key != self._clock_key:
                self._clock_key = key
                rem = self._secs(ec.get("Remaining"))
                if rem is not None:
                    self.clock = (state.t, rem, bool(ec.get("Extrapolating")))
        part = (state.topics.get("TimingData") or {}).get("SessionPart")
        try:
            part = int(part)
        except (TypeError, ValueError):
            return
        self.part_start_t.setdefault(part, state.t)
        tl = self.time_left(state.t)
        if tl is not None:
            self.part_len[part] = max(self.part_len.get(part, 0.0), tl)

    def time_left(self, t: float) -> float | None:
        if self.clock is None:
            return None
        t0, rem, running = self.clock
        return max(0.0, rem - (t - t0)) if running else rem

    def mark(self, key: tuple, t: float) -> float:
        """The first time condition ``key`` was seen true (stable however often it is asked)."""
        return self.onset.setdefault(key, t)

    def sample_cut(self, part: int, t: float, cut: float | None) -> None:
        if cut is None:
            return
        h = self.cut_hist.setdefault(part, [])
        if not h or h[-1][1] != cut:
            h.append((t, cut))

    def cut_ago(self, part: int, t: float, seconds: float) -> float | None:
        """The cut time as it was ``seconds`` before ``t`` (None if not yet known then)."""
        out = None
        for ht, c in self.cut_hist.get(part, ()):
            if ht <= t - seconds:
                out = c
        return out


def read(state: RaceState, tracker: QualiTracker) -> Picture | None:
    td = state.topics.get("TimingData") or {}
    try:
        part = int(td.get("SessionPart"))
    except (TypeError, ValueError):
        return None
    lines = td.get("Lines") or {}
    entries = tuple(int(x) for x in td.get("NoEntries") or DEFAULT_ENTRIES)
    flagged = [n for n, ln in lines.items() if isinstance(ln, dict) and ln.get("KnockedOut") is True]
    elig = [n for n, ln in lines.items() if isinstance(ln, dict) and ln.get("KnockedOut") is not True]
    bests = {n: _best(lines[n], part) for n in elig}
    pos = {n: (state.drivers[n].position if n in state.drivers and state.drivers[n].position else 999) for n in elig}
    order = sorted(elig, key=lambda n: (bests[n] is None, bests[n] if bests[n] is not None else 0.0, pos[n], n))
    cut_pos = entries[part] if part < 3 and len(entries) > part else None
    laps_part: dict = {}
    t0 = tracker.part_start_t.get(part, 0.0)
    for r in state.laps:
        if r.t_end >= t0:
            laps_part[r.driver] = laps_part.get(r.driver, 0) + 1
    cut_time = bests[order[cut_pos - 1]] if cut_pos and len(order) >= cut_pos else None
    first_out = bests[order[cut_pos]] if cut_pos and len(order) > cut_pos else None
    cars = {}
    for i, n in enumerate(order, 1):
        b = bests[n]
        d = state.drivers.get(n)
        cars[n] = CarQ(n, b, i, bool(d and d.in_pit), laps_part.get(n, 0),
                       None if (b is None or cut_time is None) else round(b - cut_time, 3),
                       cut_pos is not None and i > cut_pos)
    tl = tracker.time_left(state.t)
    return Picture(part, entries, cut_pos, cars, order, cut_time, first_out,
                   sum(1 for n in order if bests[n] is not None), tl, tracker.part_len.get(part),
                   state.session_status == "Started" and (tl is None or tl > 0), flagged)


# --------------------------------------------------------------------------- models
_MODEL_CACHE: dict = {}


def _model() -> dict:
    if "m" not in _MODEL_CACHE:
        try:
            _MODEL_CACHE["m"] = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _MODEL_CACHE["m"] = {}
    return _MODEL_CACHE["m"]


def time_bin(tl: float) -> int:
    return sum(1 for e in TIME_BINS if tl >= e)


def anchor(pic: Picture) -> tuple:
    """(cut time now, 0); else (slowest time set so far, 1); else (None, 2)."""
    if pic.cut_time is not None:
        return pic.cut_time, 0
    times = [c.best for c in pic.cars.values() if c.best is not None]
    return (max(times), 1) if times else (None, 2)


def cut_features(pic: Picture, tracker: QualiTracker, t: float) -> dict:
    a, kind = anchor(pic)
    tl = pic.time_left or 0.0
    ago = tracker.cut_ago(pic.part, t, 120.0)
    best = min((c.best for c in pic.cars.values() if c.best is not None), default=None)
    return {
        "part": pic.part, "time_left": tl, "frac_left": tl / pic.part_len if pic.part_len else 0.0,
        "n_elig": len(pic.order), "n_timed": pic.n_timed, "anchor_kind": kind,
        "trend_120": (pic.cut_time - ago) if (ago is not None and pic.cut_time is not None) else 0.0,
        "spread": (a - best) if (a is not None and best is not None) else 0.0,
        "n_in_pit": sum(1 for c in pic.cars.values() if c.in_pit),
    }


def predict_cut(pic: Picture, sprint: bool = False) -> float | None:
    """Expected cut time when the part ends: the cut now plus the track evolution still to come.

    The evolution is a table of mean improvements by (part, seconds left), fitted on 2025
    (``pitsense quali-bench``); without a model file the cut now is the prediction.
    """
    a, kind = anchor(pic)
    if a is None or pic.cut_pos is None:
        return a
    tab = _model().get("evolution", {}).get(f"{pic.part}{'s' if sprint else ''}")
    if not tab:
        return a
    key = f"{time_bin(pic.time_left or 0.0)}{'k' if kind else ''}"
    return round(a + tab.get(key, 0.0), 3)


def ko_features(c: CarQ, pic: Picture, pred: float | None) -> list:
    tl = pic.time_left or 0.0
    rel = (c.best - pred) if (c.best is not None and pred is not None) else 3.0
    rel = max(-3.0, min(3.0, rel))
    off = (c.rank - (pic.cut_pos or c.rank)) / max(len(pic.order), 1)
    tlf = min(tl, 600.0) / 600.0
    return [rel, 1.0 if c.best is None else 0.0, off, tlf, 1.0 if c.in_pit else 0.0,
            min(c.laps_in_part, 8) / 8.0, rel * tlf]


def p_knocked_out(c: CarQ, pic: Picture, pred: float | None) -> float | None:
    """Probability the car ends the part beyond the cut line (None in the last part)."""
    if pic.cut_pos is None:
        return None
    m = _model().get("ko")
    if not m:
        return 1.0 if c.in_zone else 0.0
    z = m["intercept"] + sum(w * v for w, v in zip(m["coef"], ko_features(c, pic, pred)))
    return round(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))), 4)


# --------------------------------------------------------------------------- send now
def lap_estimates(state: RaceState, pic: Picture) -> float | None:
    """Seconds from leaving the pits to the line that starts the push: 1.3 x the best lap so far.

    (Observed out-lap times include the wait in the garage, so they are not used.)
    """
    best = min((c.best for c in pic.cars.values() if c.best is not None), default=None)
    return best * 1.3 if best else None


def guidance(state: RaceState, pic: Picture, c: CarQ, pred: float | None, n_on_track: int,
             out_s: float | None, queue: int) -> dict:
    """Send-now call for one car: status, laps still needed, slack before the window shuts.

    A car in the pits needs out-lap + push lap (2 laps). The last push may start before the clock
    reaches zero, so the window shuts when the time left is shorter than an out-lap (slack < 0).
    """
    d = state.drivers.get(c.number)
    tl = pic.time_left
    needs = c.best is None or c.in_zone or (pred is not None and c.best > pred)
    if pic.cut_pos is None:
        needs = True  # last part: every run is for the grid
    laps = 0
    status = "SAFE"
    slack = None if (tl is None or out_s is None) else round(tl - out_s, 1)
    if d is not None and not d.in_pit:
        status = "ON_TRACK"
        laps = 1 if d.last_out_lap == d.laps + 1 else 0  # out-lap: the push is still to come
    elif needs:
        laps = 2
        if slack is not None and slack < 0:
            status = "TOO_LATE"
        elif slack is not None and slack <= 25.0 + 6.0 * queue + (15.0 if n_on_track >= 8 else 0.0):
            status = "SEND_NOW"
        else:
            status = "WAIT"
    return {"status": status, "laps_needed": laps, "needs_run": needs, "slack_s": slack}

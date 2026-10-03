"""Pit loss: seconds a stop costs relative to cars that stayed out.

v0.1 (``measure_stops``, used by the base benchmark features and history priors):

loss = (in-lap + out-lap of the stopping car)
       - (field median in-lap + out-lap for the same lap numbers, cars not stopping)
       - 2 * (car's pace offset to the field over its previous clean laps)

The pit-stop engineer's measurement (``measure_stop``) keeps that formula under
green flags and compares cars over the same stretch of *time* when a stop touches
a safety car or VSC; ``loss_prior`` / ``current_losses`` turn past races and this
race's stops into the expected loss now; ``CoStopRates`` and ``expected_rejoin``
answer where a car would come out when other cars may be stopping too.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
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


# ----------------------------------------------------------------------------- measured by time
# v0.1 compares a stop with the field on the *same lap numbers*. Under green flags
# that is the right comparison (and the most consistent one we have), but cars
# reach a lap number at different times: when a safety car or VSC starts or ends -
# at one moment for everyone - the field's laps hold a different share of slow
# running than the stopping car's, and the field closes up behind the safety car.
# So a stop that touches SC/VSC is measured over the *same stretch of time*: each
# reference car's two laps start at its line crossing nearest to the moment the
# stopping car started its in-lap, only cars close on the road count, and the
# medians ahead and behind are averaged so the closing-up trend cancels (cars
# behind read high, cars ahead read low).
#
# Every input must have been published by ``available_at`` (out-lap end + SETTLE_S),
# so a stop's value never depends on when it is computed.

SETTLE_S = 30.0  # reference cars get this long after the out-lap to finish their laps
NEAR_S = 15.0  # under SC/VSC, reference cars within this many seconds on the road
MIN_FIELD = 5  # cars needed for a field median on a lap
RANGES = {"green": (5.0, 45.0), "sc": (0.0, 40.0), "vsc": (0.0, 40.0), "mixed": (0.0, 45.0)}
CONDITIONS = ("green", "sc", "vsc", "mixed")


def status_at(status_log: Iterable[tuple[float, str]], t: float) -> str:
    """Track status in force at ``t`` (``RaceState.status_log`` lists the changes; green before any)."""
    code = "1"
    for ts, c in status_log:
        if ts > t:
            break
        code = c
    return code


def lane_condition(status_log: Iterable[tuple[float, str]], t_in: float, t_out: float) -> str | None:
    """Track condition while a car was in the pit lane: green | sc | vsc | mixed (None: red flag).

    Local yellows count as green; ``mixed`` means the status changed during the passage.
    """
    log = list(status_log)
    codes = {status_at(log, t_in)} | {c for ts, c in log if t_in < ts <= t_out}
    if "5" in codes:
        return None
    codes = {"1" if c == "2" else c for c in codes}
    if codes == {"1"}:
        return "green"
    if codes == {"4"}:
        return "sc"
    if codes <= {"6", "7"}:
        return "vsc"
    return "mixed"


@dataclass(frozen=True)
class StopLoss:
    """One stop's measured time loss."""

    driver: str
    in_lap: int
    loss: float  # seconds lost vs staying out
    condition: str  # green | sc | vsc | mixed: track status during the pit-lane passage
    method: str  # "laps" (field on the same lap numbers) or "time" (cars over the same time window)
    n_refs: int  # reference cars (method "time") or cars in the smaller field median ("laps")
    in_t: float  # pit entry, session clock
    lane_time: float | None  # pit entry -> exit from the timing flags
    available_at: float  # every input was published by then

    @property
    def typical(self) -> bool:
        """In the plausible range: not a slow stop, a penalty served or damage."""
        lo, hi = RANGES[self.condition]
        return lo <= self.loss <= hi


def _known(lap: LapRecord | None, cutoff: float | None) -> bool:
    """Completed and timed by ``cutoff`` (None: no cutoff)."""
    if lap is None or lap.lap_time is None:
        return False
    if cutoff is None:
        return True
    return lap.t_end <= cutoff and (lap.lap_time_t is None or lap.lap_time_t <= cutoff)


def _field_median_by(idx: LapIndex, lap_no: int, exclude: str, cutoff: float | None) -> float | None:
    times = [
        x.lap_time for x in idx.by_lap.get(lap_no, [])
        if x.driver != exclude and not x.is_in_lap and not x.is_out_lap and _known(x, cutoff)
    ]
    return median(times) if len(times) >= MIN_FIELD else None


def lap_number_loss(idx: LapIndex, p: PitEvent, cutoff: float | None = None) -> float | None:
    """v0.1's measurement (same lap numbers, pace offset to the field) from laps published by ``cutoff``."""
    laps = idx.by_driver.get(p.driver, {})
    li, lo = laps.get(p.in_lap), laps.get(p.out_lap) if p.out_lap is not None else None
    if not (_known(li, cutoff) and _known(lo, cutoff)):
        return None
    f_in = _field_median_by(idx, p.in_lap, p.driver, cutoff)
    f_out = _field_median_by(idx, p.out_lap, p.driver, cutoff)
    if f_in is None or f_out is None:
        return None
    offsets = []
    for k in range(p.in_lap - 1, max(1, p.in_lap - 6), -1):
        lap = laps.get(k)
        fm = _field_median_by(idx, k, p.driver, cutoff)
        if _clean(lap) and fm is not None:
            offsets.append(lap.lap_time - fm)
        if len(offsets) == 3:
            break
    offset = median(offsets) if offsets else 0.0
    return li.lap_time + lo.lap_time - f_in - f_out - 2 * offset


def _pace(laps: dict[int, LapRecord], before: int, n: int = 3, lookback: int = 6) -> float | None:
    """Median of up to ``n`` clean laps among the ``lookback`` laps before lap ``before``."""
    times = [laps[k].lap_time for k in range(before - 1, max(1, before - 1 - lookback), -1) if _clean(laps.get(k))]
    return median(times[:n]) if times else None


def reference_losses(
    idx: LapIndex, p: PitEvent, *, cutoff: float | None = None, max_dt: float = 60.0, pace_adjust: bool = True,
) -> list[tuple[float, float]]:
    """(time offset, loss) against each car that ran two laps without stopping over the same time window.

    The window starts when the stopping car started its in-lap. Each reference car's
    two laps start at its line crossing nearest to that moment (``dt`` = its crossing
    minus ours: positive = it was behind on the road). With ``pace_adjust`` the
    difference in recent clean-lap pace is taken out (twice: two laps).
    """
    mine = idx.by_driver.get(p.driver, {})
    li, lo = mine.get(p.in_lap), mine.get(p.out_lap) if p.out_lap is not None else None
    if p.in_lap < 2 or not (_known(li, cutoff) and _known(lo, cutoff)) or li.t_start is None:
        return []
    a, spent = li.t_start, li.lap_time + lo.lap_time
    my_pace = _pace(mine, p.in_lap) if pace_adjust else None
    out = []
    for drv, laps in idx.by_driver.items():
        if drv == p.driver:
            continue
        best, best_dt = None, max_dt
        for k, x in laps.items():
            if x.t_start is not None and abs(x.t_start - a) <= best_dt and (cutoff is None or x.t_end <= cutoff):
                best, best_dt = k, abs(x.t_start - a)
        if best is None or best < 2:
            continue
        r1, r2 = laps[best], laps.get(best + 1)
        if not (_known(r1, cutoff) and _known(r2, cutoff)) or any(r.is_in_lap or r.is_out_lap for r in (r1, r2)):
            continue
        loss = spent - r1.lap_time - r2.lap_time
        if my_pace is not None:
            ref_pace = _pace(laps, best)
            if ref_pace is not None:
                loss -= 2 * (my_pace - ref_pace)
        out.append((r1.t_start - a, loss))
    return out


def combine(refs: list[tuple[float, float]], near_s: float = NEAR_S) -> float | None:
    """One loss from reference cars within ``near_s`` on the road: mean of the medians ahead and behind."""
    near = [(dt, x) for dt, x in refs if abs(dt) <= near_s]
    ahead = [x for dt, x in near if dt < 0]
    behind = [x for dt, x in near if dt >= 0]
    if ahead and behind:
        return (median(ahead) + median(behind)) / 2
    if ahead or behind:
        return median(ahead or behind)
    return None


def measure_stop(
    idx: LapIndex, p: PitEvent, status_log: Iterable[tuple[float, str]] = (), *,
    settle_s: float = SETTLE_S, near_s: float = NEAR_S,
) -> StopLoss | None:
    """Measure one completed stop (None if it can't be measured).

    Green throughout (in-lap, pit lane, out-lap): v0.1's lap-number method. Otherwise,
    or if the field is too thin for it: reference cars over the same time window.
    """
    if p.out_lap is None or p.out_t is None or p.under_red or p.in_lap < 2:
        return None
    laps = idx.by_driver.get(p.driver, {})
    li, lo = laps.get(p.in_lap), laps.get(p.out_lap)
    if li is None or lo is None:
        return None
    cond = lane_condition(status_log, p.in_t, p.out_t)
    if cond is None:
        return None
    cutoff = lo.t_end + settle_s
    if cond == "green" and condition_of(li, lo) == "green":
        loss = lap_number_loss(idx, p, cutoff)
        if loss is not None:
            n = min(len(idx.by_lap.get(p.in_lap, [])), len(idx.by_lap.get(p.out_lap, [])))
            return StopLoss(p.driver, p.in_lap, round(loss, 3), cond, "laps", n, p.in_t, p.lane_time, cutoff)
    refs = reference_losses(idx, p, cutoff=cutoff, max_dt=near_s, pace_adjust=cond != "sc")
    loss = combine(refs, near_s)
    if loss is None:
        return None
    return StopLoss(p.driver, p.in_lap, round(loss, 3), cond, "time", len(refs), p.in_t, p.lane_time, cutoff)


def measure_stops_timed(
    idx: LapIndex, pit_events: Iterable[PitEvent], status_log: Iterable[tuple[float, str]] = (), **kw,
) -> list[StopLoss]:
    """Every completed stop that can be measured (see :func:`measure_stop`)."""
    log = list(status_log)
    return [m for p in pit_events if (m := measure_stop(idx, p, log, **kw)) is not None]


# ----------------------------------------------------------------------------- what past races teach
# A finished race is summarised as JSON (``race_summary``, stored by the pit-stop
# engineer's ``summarize_race``): its measured stops, team stationary times, and how
# often cars close behind a stopping car stopped in the same window. ``loss_prior``
# turns the races that ended before this one started into the expected loss by
# track status; ``current_losses`` updates it with this race's stops so far.

DEFAULT_RATIO = {"sc": 0.75, "vsc": 0.62}  # loss / the race's green loss, until history has enough stops
DEFAULT_SD = {"green": 2.5, "sc": 5.0, "vsc": 4.0}  # spread of one stop around the estimate (s)
K_GREEN, K_CIRCUIT, K_INRACE = 3.0, 3.0, 2.0  # weight of each prior, in stops
STATIONARY_RANGE = (1.5, 8.0)  # a stationary time outside this was a problem, not the crew's pace
GAP_WINDOW_S = 40.0  # cars this close behind a stopping car are counted for the co-stop rates


def _pss_by_stop(pit_stops: Iterable) -> dict:
    """Official pit-stop records by (driver, lap); PitStopSeries wins over the older topic."""
    out: dict = {}
    for x in pit_stops:
        key = (x.driver, x.lap)
        if key not in out or x.source == "PitStopSeries":
            out[key] = x
    return out


def race_summary(final) -> dict:
    """What a finished race teaches about pit stops (JSON values). ``final`` is the complete RaceState."""
    idx = LapIndex(final.laps)
    pss = _pss_by_stop(final.pit_stops)
    team = {n: d.team for n, d in final.drivers.items()}
    stops = []
    for m in measure_stops_timed(idx, final.pit_events, final.status_log):
        rec = pss.get((m.driver, m.in_lap))
        stops.append({
            "driver": m.driver, "team": team.get(m.driver, ""), "lap": m.in_lap, "cond": m.condition,
            "method": m.method, "loss": m.loss, "typical": m.typical,
            "lane": rec.lane_time if rec is not None else None,
            "stationary": rec.stop_time if rec is not None else None,
        })
    stationary: dict[str, list[float]] = defaultdict(list)
    for x in final.pit_stops:
        if x.stop_time is not None and STATIONARY_RANGE[0] < x.stop_time < STATIONARY_RANGE[1]:
            stationary[team.get(x.driver, "")].append(x.stop_time)
    return {"stops": stops, "stationary": dict(stationary), "costop": costop_counts(final)}


def _green(summary: dict) -> list[float]:
    return [s["loss"] for s in summary.get("stops", ()) if s["cond"] == "green" and s["typical"]]


@dataclass(frozen=True)
class LossPrior:
    """Expected pit loss before this race's own stops are counted."""

    green: float
    ratio: dict = field(default_factory=lambda: dict(DEFAULT_RATIO))  # SC/VSC loss as a share of green
    circuit: dict = field(default_factory=lambda: {"sc": (), "vsc": ()})  # typical SC/VSC losses at this circuit
    sd: dict = field(default_factory=lambda: dict(DEFAULT_SD))  # one stop around the estimate
    sd_green_prior: float = 1.5  # how far a race's green loss lands from its prior
    drift: float = 0.0  # season drift added to the circuit's last green loss
    lane_time: float | None = None  # typical green pit-lane time here (entry to exit line)
    source: str = "default"


def loss_prior(past: list[tuple[int | None, dict]], circuit_key: int | None, *,
               pool_n: int = 10, drift_n: int = 8) -> LossPrior:
    """Loss prior from finished races: ``past`` is (circuit key, ``race_summary``), oldest first.

    Green: the circuit's last visit (median of its typical green stops) plus a season
    drift - how far the latest races landed from their own circuits' last visits -
    else the median of the latest ``pool_n`` races. SC/VSC: a share of the green loss
    (pooled over every past stop), later blended with this circuit's own SC/VSC stops.
    """

    def last_visit(j: int, ck: int | None) -> float | None:
        if ck is None:
            return None
        for k in range(j - 1, -1, -1):
            if past[k][0] == ck:
                g = _green(past[k][1])
                if len(g) >= 3:
                    return median(g)
        return None

    n = len(past)
    base = last_visit(n, circuit_key)
    resid = []
    for j in range(max(0, n - drift_n), n):
        g, b = _green(past[j][1]), last_visit(j, past[j][0])
        if len(g) >= 3 and b is not None:
            resid.append(median(g) - b)
    drift = median(resid) * len(resid) / (len(resid) + 1.0) if resid else 0.0
    pooled = [x for _, s in past[-pool_n:] for x in _green(s)]
    if base is not None:
        green, source = base + drift, "circuit+drift" if resid else "circuit"
    elif len(pooled) >= 10:
        green, source = median(pooled), "global"
    else:
        green, source = DEFAULTS["green"], "default"

    ratios: dict[str, list[float]] = {"sc": [], "vsc": []}
    dev: dict[str, list[float]] = {"green": [], "sc": [], "vsc": []}
    for _, s in past:
        g = _green(s)
        if len(g) >= 3:
            gm = median(g)
            dev["green"] += [abs(x - gm) for x in g]
            for c in ("sc", "vsc"):
                ratios[c] += [x["loss"] / gm for x in s["stops"] if x["typical"] and x["cond"] == c]
    ratio = {c: median(v) if len(v) >= 5 else DEFAULT_RATIO[c] for c, v in ratios.items()}
    for _, s in past:
        g = _green(s)
        if len(g) >= 3:
            gm = median(g)
            for c in ("sc", "vsc"):
                dev[c] += [abs(x["loss"] - ratio[c] * gm) for x in s["stops"] if x["typical"] and x["cond"] == c]
    sd = {c: 1.4826 * median(v) if len(v) >= 10 else DEFAULT_SD[c] for c, v in dev.items()}
    sd_prior = 1.4826 * median(abs(r - median(resid)) for r in resid) if len(resid) >= 5 else 1.5
    same = [s for ck, s in past if circuit_key is not None and ck == circuit_key]
    circuit = {c: tuple(x["loss"] for s in same for x in s.get("stops", ()) if x["typical"] and x["cond"] == c)
               for c in ("sc", "vsc")}
    lanes = [x["lane"] for s in same[-1:] for x in s.get("stops", ())
             if x["cond"] == "green" and x["typical"] and x.get("lane")]
    return LossPrior(round(green, 3), ratio, circuit, sd, round(sd_prior, 3), round(drift, 3),
                     round(median(lanes), 3) if lanes else None, source)


def current_losses(prior: LossPrior, samples: dict[str, list[float]]) -> dict[str, float]:
    """Expected loss by track status now: the prior updated with this race's measured stops.

    The green estimate moves with this race's green stops; SC and VSC follow it through
    their ratio, then move toward this circuit's past SC/VSC stops and this race's own.
    """
    green = shrink(prior.green, samples.get("green", []), K_GREEN)
    out = {"green": green}
    for c in ("sc", "vsc"):
        circuit = shrink(prior.ratio[c] * green, list(prior.circuit.get(c, ())), K_CIRCUIT)
        out[c] = shrink(circuit, samples.get(c, []), K_INRACE)
    return out


def loss_sd(prior: LossPrior, condition: str, n_green: int = 0) -> float:
    """Spread of one stop's loss around ``current_losses`` (green also carries the prior's own error)."""
    if condition == "green":
        est = prior.sd_green_prior * math.sqrt(K_GREEN / (K_GREEN + n_green))
        return math.hypot(prior.sd["green"], est)
    return prior.sd[condition]


# ----------------------------------------------------------------------------- cars stopping together
# Under green flags a car behind rarely stops in the same window (~12%); under a
# safety car most do (~60% of those that haven't stopped yet), and nobody stops
# twice. Rates are counted in finished races by condition, whether the car already
# stopped in this phase, its tyre age and how long the SC/VSC has been out.

COSTOP_DEFAULT = {"green": 0.12, "sc": 0.6, "vsc": 0.4}
TYRE_BINS = (3, 8, 15, 25)  # upper edges of the tyre-age bins (laps on the set)
PHASE_BINS = (60.0, 150.0)  # SC/VSC age bins (s since the status started)
ALREADY_S = 300.0  # a stop in this phase and in the last 5 minutes = already stopped


def costop_key(cond: str, already: bool, tyre_age: int | None, phase_age: float) -> str:
    tyre = "u" if tyre_age is None else str(sum(1 for b in TYRE_BINS if tyre_age > b))
    age = str(sum(1 for b in PHASE_BINS if phase_age > b)) if cond in ("sc", "vsc") else "-"
    return f"{cond}|{int(already)}|{tyre}|{age}"


def pass_through(rc: Iterable, track_status: str, since: float) -> bool:
    """Race control has the whole field driving through the pit lane behind the safety car.

    Starts with "... THROUGH THE PIT LANE", ends with "... USE START/FINISH STRAIGHT" or the SC.
    ``rc`` holds objects with ``t`` and ``message`` (``RaceState.rc``).
    """
    if track_status != "4":
        return False
    on = False
    for m in rc:
        if m.t < since:
            continue
        text = m.message.upper()
        if "THROUGH THE PIT LANE" in text:
            on = True
        elif "START/FINISH STRAIGHT" in text and "USE" in text:
            on = False
    return on


def condition_now(track_status: str) -> str | None:
    """green | sc | vsc for a track status code (None: red flag)."""
    return {"4": "sc", "6": "vsc", "7": "vsc", "5": None}.get(track_status, "green")


def costop_counts(final) -> dict[str, list[int]]:
    """In a finished race: for cars close behind each stopping car, did they stop in its window too?

    Window: from 5 s before the stop to the end of the stopping car's out-lap. Cars
    behind are those behind on the timing screen at the line before the in-lap.
    """
    laps = LapIndex(final.laps)
    log = list(final.status_log)
    events = [p for p in final.pit_events if not p.under_red]
    by_driver: dict[str, list[float]] = defaultdict(list)
    for p in events:
        by_driver[p.driver].append(p.in_t)
    counts: dict[str, list[int]] = {}
    for p in events:
        mine = laps.by_driver.get(p.driver, {})
        before = mine.get(p.in_lap - 1)
        if before is None or before.position is None:
            continue
        status = status_at(log, p.in_t)
        cond = condition_now(status)
        if cond is None:
            continue
        since = max([t for t, _ in log if t <= p.in_t], default=0.0)
        if cond == "sc" and pass_through([m for m in final.rc if m.t <= p.in_t], status, since):
            continue  # nobody really stops: the field drives through the pit lane
        out = mine.get(p.out_lap) if p.out_lap is not None else None
        end = out.t_end if out is not None else p.in_t + 150.0
        for x in laps.by_lap.get(p.in_lap - 1, []):
            if x.driver == p.driver or x.position is None or x.position <= before.position:
                continue
            gap = x.t_end - before.t_end
            if not 0 < gap <= GAP_WINDOW_S:
                continue
            already = any(max(since, p.in_t - ALREADY_S) <= t < p.in_t - 5.0 for t in by_driver[x.driver])
            stopped = any(p.in_t - 5.0 <= t <= end for t in by_driver[x.driver])
            key = costop_key(cond, already, x.tyre_age, p.in_t - since)
            n, k = counts.get(key, (0, 0))
            counts[key] = [n + 1, k + int(stopped)]
    return counts


class CoStopRates:
    """P(a car also stops in this window) from counts of finished races.

    Smoothed in levels: a bin moves toward its (condition, already stopped) rate, that
    toward the condition's rate, and that toward ``COSTOP_DEFAULT``.
    """

    def __init__(self, counts: dict[str, list[int]] | None = None, *,
                 m_key: float = 5.0, m_already: float = 10.0, m_cond: float = 20.0):
        self.counts = dict(counts or {})
        self.m_key = m_key

        def total(prefix: str) -> tuple[int, int]:
            rows = [v for k, v in self.counts.items() if k.startswith(prefix)]
            return sum(v[0] for v in rows), sum(v[1] for v in rows)

        self.base: dict[str, float] = {}
        for c, default in COSTOP_DEFAULT.items():
            n, s = total(c + "|")
            cond_rate = (s + m_cond * default) / (n + m_cond)
            for already in (0, 1):
                n, s = total(f"{c}|{already}|")
                self.base[f"{c}|{already}"] = (s + m_already * cond_rate) / (n + m_already)

    @staticmethod
    def merged(tables: Iterable[dict[str, list[int]]]) -> "CoStopRates":
        total: dict[str, list[int]] = {}
        for t in tables:
            for k, (n, s) in t.items():
                a, b = total.get(k, (0, 0))
                total[k] = [a + n, b + s]
        return CoStopRates(total)

    def rate(self, cond: str, already: bool, tyre_age: int | None, phase_age: float) -> float:
        base = self.base.get(f"{cond}|{int(already)}", COSTOP_DEFAULT["green"])
        n, s = self.counts.get(costop_key(cond, already, tyre_age, phase_age), (0, 0))
        return (s + self.m_key * base) / (n + self.m_key)


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass(frozen=True)
class Rejoin:
    position: int  # expected position after the stop
    passers: float  # expected cars that get by (not stopping themselves)
    within: int  # cars behind within one loss, ignoring who else stops
    within_stopping: float  # of those, how many are expected to stop too
    gap_ahead: float | None  # s behind the car we'd come out behind (None: nobody)
    gap_behind: float | None  # s ahead of the first car we'd stay ahead of (None: nobody)


REJOIN_STAT = "mode"  # which summary of the passers' distribution is the expected position: mean | median | mode


def _pmf(qs: list[float]) -> list[float]:
    """Distribution of the number of successes among independent chances ``qs`` (Poisson binomial)."""
    pmf = [1.0]
    for q in qs:
        nxt = [0.0] * (len(pmf) + 1)
        for k, v in enumerate(pmf):
            nxt[k] += v * (1.0 - q)
            nxt[k + 1] += v * q
        pmf = nxt
    return pmf


def _summary(pmf: list[float], stat: str) -> int:
    if stat == "mode":
        return max(range(len(pmf)), key=lambda k: (pmf[k], -k))
    if stat == "median":
        acc = 0.0
        for k, v in enumerate(pmf):
            acc += v
            if acc >= 0.5:
                return k
    return math.floor(sum(k * v for k, v in enumerate(pmf)) + 0.5)


def expected_rejoin(position: int, gaps: list[float], loss: float, sd: float, p_stop: list[float],
                    stat: str | None = None) -> Rejoin:
    """Where a car stopping now comes out.

    ``gaps``: seconds to each car behind in running order (cumulative); ``p_stop``: the
    chance each of them stops in the same window. A car behind gets by if it doesn't
    stop and its gap is under the loss, with the loss uncertain by ``sd`` (normal).
    The position is the median (``stat``) of the resulting distribution of passers.
    """
    passers, within, within_stop = 0.0, 0, 0.0
    gap_ahead = gap_behind = None
    qs: list[float] = []
    for g, p in zip(gaps, p_stop):
        z = (loss - g) / sd if sd > 0 else (math.inf if g < loss else -math.inf)
        if z < -4.0:
            if gap_behind is None and p < 0.5:
                gap_behind = g - loss
            break
        q = _phi(z) * (1.0 - p)
        qs.append(q)
        passers += q
        if g < loss:
            within += 1
            within_stop += p
            if p < 0.5:
                gap_ahead = loss - g
        elif gap_behind is None and p < 0.5:
            gap_behind = g - loss
    n = _summary(_pmf(qs), stat or REJOIN_STAT)
    return Rejoin(position + n, round(passers, 3), within, round(within_stop, 3),
                  None if gap_ahead is None else round(gap_ahead, 3),
                  None if gap_behind is None else round(gap_behind, 3))

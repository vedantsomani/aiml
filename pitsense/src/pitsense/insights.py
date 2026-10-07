"""Analysis panels on any race state, live or recorded, as of now: tyre degradation, track evolution, two drivers
compared, and a lap's speed trace. Everything reads ``RaceState`` (laps and feeds already published), never later.

Clean laps (``clean``): a lap time, not lap 1, not an in- or out-lap, green flag the whole lap, and at least
``MIN_GAP_AHEAD`` s to the car ahead (or leading). Lap times are fuel-corrected with a simple linear term
(``FUEL_S_PER_LAP``: a lap later is that much quicker on fuel alone). The degradation fit is behind ``DegFit`` so a
state estimator (Kalman) can replace the linear fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Protocol

import numpy as np

FUEL_S_PER_LAP = 0.055  # s/lap gained as fuel burns (simple linear term)
MIN_GAP_AHEAD = 1.0  # s: closer than this, the car ahead's air spoils the lap
BAND_Q = (0.10, 0.90)  # 80 % band


def clean(r) -> bool:
    return (r.lap_time is not None and r.lap > 1 and not r.is_in_lap and not r.is_out_lap and r.track_status == "1"
            and not r.lap_time_inferred and (r.position == 1 or (r.interval is not None and r.interval >= MIN_GAP_AHEAD)))


def fuel_corrected(r) -> float:
    return float(r.lap_time) + FUEL_S_PER_LAP * r.lap


# ----------------------------------------------------------------------------- degradation
@dataclass
class Fit:
    slope: float  # s per lap of tyre age (degradation rate)
    intercept: float
    lo: float  # 80 % band: offsets from the line
    hi: float
    n: int


class DegFit(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> Fit | None: ...


class LinearDeg:
    """Least squares line; the band is the 10th-90th percentile of the residuals (needs ``min_n`` laps)."""

    min_n = 4

    def fit(self, x: np.ndarray, y: np.ndarray) -> Fit | None:
        if len(x) < self.min_n or np.ptp(x) == 0:
            return None
        slope, icpt = np.polyfit(x, y, 1)
        res = y - (slope * x + icpt)
        lo, hi = np.quantile(res, BAND_Q)
        return Fit(float(slope), float(icpt), float(lo), float(hi), len(x))


def tyre_deg(state, cars: list[str] | None = None, fitter: DegFit | None = None) -> list[dict]:
    """Per car and stint: clean fuel-corrected lap time against tyre age, the fitted line, its 80 % band and the
    degradation rate (s/lap)."""
    fitter = fitter or LinearDeg()
    by: dict[tuple, list] = {}
    for r in state.laps:
        if (cars is None or r.driver in cars) and clean(r) and r.tyre_age is not None:
            by.setdefault((r.driver, r.stint), []).append(r)
    out = []
    for (car, stint), rs in sorted(by.items(), key=lambda kv: (int(kv[0][0]) if kv[0][0].isdigit() else 999, kv[0][1])):
        x = np.array([r.tyre_age for r in rs], float)
        y = np.array([fuel_corrected(r) for r in rs])
        f = fitter.fit(x, y)
        d = state.drivers.get(car)
        out.append({"car": car, "tla": d.tla if d else car, "stint": stint, "compound": rs[-1].compound,
                    "points": [[int(a), round(float(b), 3)] for a, b in zip(x, y)],
                    "deg_s_per_lap": None if f is None else round(f.slope, 4),
                    "line": None if f is None else [round(f.intercept, 3), round(f.slope, 4)],
                    "band": None if f is None else [round(f.lo, 3), round(f.hi, 3)], "n": len(rs)})
    return out


# ----------------------------------------------------------------------------- track evolution
def track_evolution(state) -> dict:
    """Median clean fuel-corrected lap time per race lap, and the fitted evolution (s per lap, negative = faster)."""
    by: dict[int, list[float]] = {}
    for r in state.laps:
        if clean(r):
            by.setdefault(r.lap, []).append(fuel_corrected(r))
    laps = sorted(k for k, v in by.items() if len(v) >= 3)
    med = [round(median(by[k]), 3) for k in laps]
    slope = float(np.polyfit(laps, med, 1)[0]) if len(laps) >= 5 else None
    return {"laps": laps, "median_s": med, "evo_s_per_lap": None if slope is None else round(slope, 4)}


# ----------------------------------------------------------------------------- two drivers
def compare(state, a: str, b: str) -> dict:
    """Two drivers: lap times per lap, the gap at each line crossing (positive: ``a`` behind) and per-stint pace delta
    (``a``'s median clean lap minus ``b``'s over the same laps)."""
    la = {r.lap: r for r in state.laps if r.driver == a}
    lb = {r.lap: r for r in state.laps if r.driver == b}
    laps = sorted(set(la) | set(lb))
    times = {"a": [la[k].lap_time if k in la else None for k in laps], "b": [lb[k].lap_time if k in lb else None for k in laps]}
    gap = [round(la[k].t_end - lb[k].t_end, 3) if k in la and k in lb else None for k in laps]
    stints = []
    for s in sorted({r.stint for r in la.values()}):
        ks = [k for k, r in la.items() if r.stint == s]
        ca = [fuel_corrected(la[k]) for k in ks if clean(la[k])]
        cb = [fuel_corrected(lb[k]) for k in ks if k in lb and clean(lb[k])]
        stints.append({"stint": s, "laps": [min(ks), max(ks)], "compound": la[max(ks)].compound,
                       "delta_s": round(median(ca) - median(cb), 3) if ca and cb else None, "n": [len(ca), len(cb)]})
    tla = {n: (state.drivers[n].tla if n in state.drivers else n) for n in (a, b)}
    return {"a": a, "b": b, "tla": tla, "laps": laps, "lap_time": times, "gap_s": gap, "stints": stints}


# ----------------------------------------------------------------------------- speed trace
def lap_samples(state, car: str, lap: int) -> dict | None:
    """CarData of one lap on a distance axis (m), from the published samples: time aligned through Position.
    None if the lap is not in the feeds' window (about ten minutes are kept) or there is no telemetry."""
    rec = next((r for r in state.laps if r.driver == car and r.lap == lap), None)
    store = state.feeds.telemetry
    if rec is None or rec.t_start is None:
        return None
    try:
        tel, pos = store.telemetry(car, None), store.position_history(car, None)
    except Exception:
        return None
    if tel is None or pos is None or len(tel["t"]) < 10 or len(pos["t"]) < 10:
        return None
    off = float(np.min(tel["t"] - tel["utc"]))  # clock offset: the quickest publication
    poff = float(np.min(pos["t"] - pos["utc"]))
    ts, ps = tel["utc"] + off, pos["utc"] + poff
    k = (ts >= rec.t_start) & (ts <= rec.t_end)
    j = (ps >= rec.t_start - 1) & (ps <= rec.t_end + 1)
    if k.sum() < 10 or j.sum() < 5:
        return None
    px, py, pt = pos["x"][j], pos["y"][j], ps[j]
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))]) / 10.0  # feed units are 1/10 m
    t = ts[k]
    d = np.interp(t, pt, dist) - np.interp(rec.t_start, pt, dist)
    return {"car": car, "lap": lap, "t": t - t[0], "dist": d, "x": np.interp(t, pt, px), "y": np.interp(t, pt, py),
            "speed": tel["speed"][k], "throttle": tel["throttle"][k],
            "brake": tel["brake"][k], "gear": tel["gear"][k], "lap_time": rec.lap_time}


def speed_trace(state, car: str, lap: int) -> dict | None:
    s = lap_samples(state, car, lap)
    if s is None:
        return None
    return {"car": car, "lap": lap, "lap_time": s["lap_time"], "dist": [round(float(x), 1) for x in s["dist"]],
            "speed": [None if v != v else round(float(v)) for v in s["speed"]]}

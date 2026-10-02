"""Tyre & performance engineer: how fast each car is on its tyres, and how fast that fades.

Per car: ``pace_s`` (next clean lap), ``deg_s_per_lap`` (wear, fuel removed),
``fresh_<compound>_s`` (first flying lap on a new set fitted now), ``cliff_risk``,
plus uncertainty and traffic. Per race: ``fuel_s_per_lap`` and the field's compound
offsets and wear rates. ``model.py`` has the maths.

* ``observe`` indexes new laps and, every ``REFIT_EVERY`` new laps, refits the field
  model (fuel trend, compound offsets, wear by compound) on all clean laps so far.
* ``car`` filters the car's pace over its clean laps and takes its wear from the current
  set. Results are cached until the car's laps, its tyres or the field fit change (a lap
  time can still change after the lap appears: a late LastLapTime replaces an inferred
  one, so the latest lap's time is part of the cache key).

Clean lap = timed, not lap 1, not an in/out-lap, green throughout (``memory.is_clean``),
on a dry compound, and within ``RACING`` of the field's ``RACING_RANK``-th fastest clean
lap so far: some neutralised laps are published under a green status (Baku 2025, lap 4:
~185 s laps against ~107 s racing laps). A lap started within ``TRAFFIC_S`` of the car
ahead counts half (``TRAFFIC_W``) in the fits.
"""

from __future__ import annotations

import bisect
import math
from statistics import median

import numpy as np

from ...engineer import Engineer
from ...memory import is_clean
from .model import CI, DRY, Priors, expected, fit_car, fit_field, prior_fit

REFIT_EVERY = 10  # new laps between field refits
TRAFFIC_S = 1.0  # a lap started within this many seconds of the car ahead may be held up
TRAFFIC_W = 0.5  # weight of such a lap in the fits
RACING = 1.07  # a clean lap slower than this times the reference isn't racing pace
RACING_RANK = 5  # the reference: the field's 5th-fastest clean lap so far


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


class TyreEngineer(Engineer):
    name = "tyre"

    def __init__(self, ctx, memory) -> None:
        super().__init__(ctx, memory)
        self.pri = Priors()
        self._n_laps = 0
        self._n_at_fit = 0
        self._stints: dict[str, dict[int, tuple[int, int, str]]] = {}  # car -> stint -> (fit lap, age, compound)
        self._cand: list = []  # laps that may be clean once their time is known (refs into the state)
        self._ff = prior_fit(self.pri)
        self._ff_version = 0
        self._cap = math.inf  # racing-pace limit, from the last refit
        self._cache: dict[str, tuple[tuple, dict]] = {}

    # ------------------------------------------------------------------ bookkeeping
    def _note(self, d) -> None:
        """Remember which set the car is on (re-anchored stints overwrite their entry)."""
        if d.stint > 0 and d.compound:
            self._stints.setdefault(d.number, {})[d.stint] = (d.stint_start_lap, d.stint_start_age, d.compound)

    def _set_of(self, number: str, lap: int) -> tuple[int, int, str] | None:
        """(stint, tyre age, compound) of a lap, from what is known now."""
        best = None
        for k, (s_lap, s_age, comp) in self._stints.get(number, {}).items():
            if s_lap < lap and (best is None or (s_lap, k) > (best[1], best[0])):
                best = (k, s_lap, s_age, comp)
        if best is None:
            return None
        k, s_lap, s_age, comp = best
        return k, s_age + lap - s_lap, comp

    def _weight(self, number: str, lap: int) -> float:
        prev = self.memory.index.by_driver.get(number, {}).get(lap - 1)
        if prev is not None and prev.interval is not None and prev.interval < TRAFFIC_S and prev.position != 1:
            return TRAFFIC_W
        return 1.0

    def _row(self, x) -> tuple | None:
        """(lap, age, compound index, time, weight, stint) of a clean racing lap, else None."""
        if not is_clean(x) or x.lap_time > self._cap:
            return None
        s = self._set_of(x.driver, x.lap)
        if s is None or s[2] not in CI:
            return None
        return x.lap, s[1], CI[s[2]], x.lap_time, self._weight(x.driver, x.lap), s[0]

    def observe(self, state) -> None:
        n = len(state.laps)
        if n == self._n_laps:
            return
        for rec in state.laps[self._n_laps:n]:
            d = state.drivers.get(rec.driver)
            if d is not None:
                self._note(d)
            if rec.lap > 1 and not rec.is_in_lap and not rec.is_out_lap and rec.track_status == "1":
                self._cand.append(rec)
        self._n_laps = n
        if n - self._n_at_fit >= REFIT_EVERY:
            self._refit()
            self._n_at_fit = n

    def _refit(self) -> None:
        fast: list[float] = []
        for rec in self._cand:
            if rec.lap_time is not None and (s := self._set_of(rec.driver, rec.lap)) is not None and s[2] in CI:
                bisect.insort(fast, rec.lap_time)
                del fast[RACING_RANK:]
        if fast:
            self._cap = RACING * fast[-1]
        rows = [(x.driver, r) for x in self._cand if (r := self._row(x)) is not None]
        if not rows:
            return
        a = np.array([r for _, r in rows], dtype=float)
        self._ff = fit_field([d for d, _ in rows], a[:, 0], a[:, 1], a[:, 2].astype(int), a[:, 3], a[:, 4],
                             a[:, 5].astype(int), self.pri)
        self._ff_version += 1

    # ------------------------------------------------------------------ values
    def car(self, state, number, view):
        d = state.drivers.get(number)
        if d is None:
            return {}
        self._note(d)
        laps = self.memory.index.by_driver.get(number, {})
        last = laps.get(d.laps)
        key = (self._ff_version, d.laps, getattr(last, "lap_time", None), getattr(last, "lap_time_inferred", None),
               d.stint, d.stint_start_lap, d.stint_start_age, d.compound, d.tyre_age)
        hit = self._cache.get(number)
        if hit is None or hit[0] != key:
            hit = (key, self._compute(d, laps))
            self._cache[number] = hit
        return dict(hit[1])

    def _compute(self, d, laps: dict) -> dict:
        stint_times = [x.lap_time for k, x in sorted(laps.items()) if k > d.stint_start_lap and is_clean(x)]
        out = {
            "stint_pace": _r(median(stint_times)) if stint_times else None,
            "stint_clean_laps": len(stint_times),
            "last_clean_s": stint_times[-1] if stint_times else None,
            "pace_s": None,
            "pace_sd_s": None,
            "pace_trend_s": None,
            "deg_s_per_lap": None,
            "deg_sd": None,
            "fresh_soft_s": None,
            "fresh_medium_s": None,
            "fresh_hard_s": None,
            "cliff_risk": None,
            "traffic": None,
            "resid_last_s": None,
            "resid_trend_s": None,
        }
        last = laps.get(d.laps)
        if last is not None and last.interval is not None and last.position != 1:
            out["traffic"] = bool(last.interval < TRAFFIC_S)
        rows = [r for k in sorted(laps) if (r := self._row(laps[k])) is not None]
        if not rows:
            return out
        a = np.array(rows, dtype=float)
        ff, pri = self._ff, self.pri
        fit = fit_car(a[:, 0], a[:, 1], a[:, 2].astype(int), a[:, 3], a[:, 4], a[:, 5].astype(int), d.stint, ff, pri)
        if fit is None:
            return out
        out["resid_last_s"] = _r(fit.resid_last)
        out["resid_trend_s"] = _r(fit.resid_trend)
        L = d.laps
        noise = ff.noise if ff.n else pri.noise
        c = CI.get(d.compound or "")
        if c is not None and d.tyre_age is not None:
            net = fit.net if fit.n_stint else ff.net[c]
            net_sd = fit.net_sd if fit.n_stint else pri.car_net_sd
            age = d.tyre_age
            out["pace_s"] = _r(expected(fit, ff, pri, c, net, age + 1, L + 1))
            out["pace_trend_s"] = _r(expected(fit, ff, pri, c, net, age + 1, L + 1, fast=False))
            out["pace_sd_s"] = _r(math.sqrt(fit.level_sd ** 2 + noise ** 2))
            out["deg_s_per_lap"] = _r(net + ff.fuel, 4)
            out["deg_sd"] = _r(math.sqrt(net_sd ** 2 + ff.fuel_sd ** 2), 4)
        for comp in DRY:
            j = CI[comp]
            # pit at the end of the next lap: out-lap L+2, first flying lap L+3 at age 2
            out[f"fresh_{comp.lower()}_s"] = _r(expected(fit, ff, pri, j, ff.net[j], 2, L + 3) + pri.first_lap)
        return out

    def race(self, state, view):
        ff = self._ff
        deg = ff.deg
        return {
            "fuel_s_per_lap": _r(ff.fuel, 4),
            "fuel_sd": _r(ff.fuel_sd, 4),
            "field_deg_soft": _r(deg[0], 4),
            "field_deg_medium": _r(deg[1], 4),
            "field_deg_hard": _r(deg[2], 4),
            "off_soft_s": _r(ff.off[0]),
            "off_hard_s": _r(ff.off[2]),
            "field_laps": ff.n,
        }

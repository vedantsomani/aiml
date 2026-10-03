"""Weather engineer: rain nowcast and the slick / intermediate crossover.

Two jobs, both as of now:

* ``rain_prob_10min``: probability that the feed's ``Rainfall`` flag is up at some
  moment in the next 10 minutes. A small logistic model over the weather trends, the
  FIA "risk of rain" figure (``rules`` engineer) and a circuit prior. It is trained on
  samples that :meth:`summarize_race` stored for every earlier race (``ctx.past_races``,
  so only races that ended before this one started).
* ``crossover``: cars on different tyres over the same clean laps. The earliest
  switchers show whether the other tyre has become faster (``to_inters``: inters beat
  slicks; ``to_slicks``: slicks beat inters).

Feed facts the model rests on: ``Rainfall`` is a 0/1 flag, sampled about once a minute.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from pathlib import Path
from statistics import mean, median

import numpy as np

from ...config import DRY_COMPOUNDS
from ..engineer import Engineer
from ..types import Alert
from .rules.parser import parse

_WET = ("INTERMEDIATE", "WET")
_FIELDS = ("Rainfall", "TrackTemp", "AirTemp", "Humidity")

HORIZON_S = 600.0  # "within 10 minutes"
GRID_S = 120.0  # spacing of the training samples stored per race
TREND_S = 600.0  # window of the temperature / humidity trends
SINCE_CAP_MIN = 90.0
RUN_CAP_MIN = 30.0
CIRCUIT_PSEUDO_RACES = 3.0  # shrinkage of a circuit's rain rate towards the global rate
MIN_TRAIN_POSITIVES = 40
ALERT_PROB = 0.35  # rain-onset alert threshold (dry now)

# crossover
WINDOW_LAPS = 4  # laps looked back
CROSS_MARGIN_S = 0.5  # the faster tyre must be at least this much faster (mean of the last two shared laps)
CROSS_MIN_LAPS = 2  # shared laps needed (both groups on track, clean laps) ...
CROSS_ONE_LAP_S = 1.5  # ... or one shared lap that differs by at least this much
CROSS_HOLD_LAPS = 8  # keep the call this long after the evidence disappears
RACING = 1.25  # clean lap slower than this times the stint reference is dropped (neutralised)


# --------------------------------------------------------------------------- weather series
class Series:
    """The weather feed as a step function: a sample is appended when a reading changes."""

    def __init__(self) -> None:
        self.t: list[float] = []
        self.v: list[tuple] = []  # (rainfall, track, air, humidity)

    def add(self, t: float, v: tuple) -> bool:
        if self.v and self.v[-1] == v:
            return False
        if all(x is None for x in v):
            return False
        self.t.append(t)
        self.v.append(v)
        return True

    def index(self, t: float) -> int:
        """Index of the latest sample at or before t (-1 if none)."""
        return bisect_right(self.t, t) - 1


def _val(v: tuple, k: int) -> float | None:
    x = v[k]
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def _raining(v: tuple) -> bool:
    return (_val(v, 0) or 0.0) > 0


def series_features(s: Series, t: float, rc_risk: float | None) -> dict[str, float]:
    """Everything the nowcast knows at ``t`` (only samples at or before ``t``). No NaN, no None."""
    i = s.index(t)
    out = {"rain_now": 0.0, "since_rain_min": -1.0, "run_min": 0.0, "hum": 0.0, "hum_tr": 0.0,
           "trk_tr": 0.0, "air_tr": 0.0, "rain_seen": 0.0, "rc_known": 0.0, "rc_risk": 0.0}
    if rc_risk is not None:
        out["rc_known"], out["rc_risk"] = 1.0, float(rc_risk) / 100.0
    if i < 0:
        return out
    cur = s.v[i]
    now = _raining(cur)
    out["rain_now"] = 1.0 if now else 0.0
    out["hum"] = (_val(cur, 3) or 0.0) / 100.0
    j0 = max(s.index(t - TREND_S), 0)
    prev = s.v[j0]
    for key, k in (("hum_tr", 3), ("trk_tr", 1), ("air_tr", 2)):
        a, b = _val(cur, k), _val(prev, k)
        out[key] = 0.0 if a is None or b is None else a - b
    # last time the flag was up, and how long the current run has lasted
    k = i
    while k >= 0 and not _raining(s.v[k]):
        k -= 1
    if k >= 0:
        out["rain_seen"] = 1.0
        if now:
            r = k
            while r - 1 >= 0 and _raining(s.v[r - 1]):
                r -= 1
            out["run_min"] = (t - s.t[r]) / 60.0
        else:
            out["since_rain_min"] = (t - s.t[k + 1]) / 60.0  # the sample that dropped the flag
    return out


def design(f: dict[str, float], prior_logit: float) -> list[float]:
    """The logistic model's inputs (scaled to about unit size)."""
    now = f["rain_now"]
    run = min(f["run_min"], RUN_CAP_MIN) / RUN_CAP_MIN
    since = f["since_rain_min"]
    recent = math.exp(-since / 20.0) if since >= 0 else 0.0
    # The FIA risk figure is deliberately not an input: fed in as published it made held-out log loss
    # worse on both 2025 and 2026 (docs/engineers/weather.md). It is still exposed as rc_rain_risk_f.
    return [
        now, now * run, (1 - now) * recent, (1 - now) * f["rain_seen"],
        f["hum_tr"] / 5.0, f["trk_tr"] / 5.0, f["air_tr"] / 3.0, f["hum"], prior_logit,
    ]


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def circuit_prior(races: list[tuple[int, int]], circuit_key, global_rate: float) -> float:
    """Shrunk share of earlier races at this circuit that had rain. ``races`` = [(circuit_key, rained)]."""
    here = [r for c, r in races if c == circuit_key]
    return (sum(here) + CIRCUIT_PSEUDO_RACES * global_rate) / (len(here) + CIRCUIT_PSEUDO_RACES)


def _rc_risk_at(rc: list[tuple[float, float]], t: float) -> float | None:
    """Latest "risk of rain" percentage published at or before t (rc is sorted by time)."""
    out = None
    for ts, v in rc:
        if ts > t:
            break
        out = v
    return out


def _read_weather(meta: dict) -> Series | None:
    """The finished race's weather feed (past races only: used when summarising them)."""
    from ...archive import SessionRef
    from ...events import parse_stream_line

    try:
        path = SessionRef.from_dict(meta).local_dir / "WeatherData.jsonStream"
        text = Path(path).read_bytes().decode("utf-8-sig")
    except Exception:
        return None
    cur: dict = {}
    s = Series()
    for line in text.splitlines():
        parsed = parse_stream_line(line)
        if parsed is None or not isinstance(parsed[1], dict):
            continue
        t, data = parsed
        for k, x in data.items():
            try:
                cur[k] = float(x)
            except (TypeError, ValueError):
                cur[k] = None
        s.add(t, tuple(cur.get(k) for k in _FIELDS))
    return s


def _compound_class(c: str | None) -> str | None:
    if c in _WET:
        return "wet"
    if c in DRY_COMPOUNDS:
        return "dry"
    return None


# --------------------------------------------------------------------------- engineer
class WeatherEngineer(Engineer):
    name = "weather"
    requires = ("rules",)
    # Declaring them (rain_prob_10min, rain_now, crossover_f, inters_vs_slicks_f; sentinels, never None)
    # made the core pit_within / rejoin models marginally worse, so none are declared: the columns
    # are in every row and the rain_10min task's models read them directly.
    features = ()

    @classmethod
    def summarize_race(cls, final, meta):
        """Per-race rain facts and training samples for the nowcast (JSON values).

        ``samples`` rows: [rain_now, since_rain_min, run_min, rc_risk(0-1), rc_known, hum,
        hum_tr, trk_tr, air_tr, rain_seen, y_10min]. The weather feed is read from the finished
        race's own stream, which is fine: only later races see this summary.
        """
        s = _read_weather(meta)
        if s is None or not s.t or final.started_t is None:
            return {}
        t0 = final.started_t
        t1 = final.finished_t if final.finished_t is not None else (final.t or t0)
        rc = []
        for m in final.rc:
            ev = parse(m.message, m.category, m.flag)
            if ev.kind == "rain_risk":
                rc.append((m.t, ev.value))
        rc.sort()
        samples = []
        rain_samples = 0
        t = t0
        while t <= t1:
            f = series_features(s, t, _rc_risk_at(rc, t))
            i = s.index(t)
            j = bisect_right(s.t, t + HORIZON_S)
            y = 1 if any(_raining(s.v[k]) for k in range(max(i, 0), j)) else 0
            samples.append([f["rain_now"], round(f["since_rain_min"], 2), round(f["run_min"], 2), round(f["rc_risk"], 3),
                            f["rc_known"], round(f["hum"], 3), round(f["hum_tr"], 2), round(f["trk_tr"], 2),
                            round(f["air_tr"], 2), f["rain_seen"], y])
            rain_samples += f["rain_now"] > 0
            t += GRID_S
        wet_laps = sum(1 for x in final.laps if x.compound in _WET)
        return {"rained": int(any(r[10] == 1 and r[0] == 1 for r in samples)),
                "rain_frac": round(rain_samples / max(len(samples), 1), 4),
                "wet_laps": wet_laps, "samples": samples}

    def __init__(self, ctx, memory):
        super().__init__(ctx, memory)
        self.series = Series()
        self._model = None
        self._prior_logit = 0.0
        self._prior_rate = 0.0
        self._fitted = False
        self._cross: dict = {"call": "none", "t": None, "seen_lap": -999, "delta": None}
        self._cross_key = None
        self._p_cache = (None, None)

    # ------------------------------------------------------------------ pre-race knowledge
    def _fit(self) -> None:
        self._fitted = True
        past = sorted((s for s in self.ctx.past_races if isinstance(s.extra.get("weather"), dict) and s.extra["weather"].get("samples")),
                      key=lambda s: s.end_utc)
        if not past:
            return
        flags = [(s.circuit_key, s.extra["weather"]["rained"]) for s in past]
        g = (sum(r for _, r in flags) + 0.5) / (len(flags) + 1.0)
        self._prior_rate = circuit_prior(flags, self.ctx.meta.get("circuit_key"), g)
        self._prior_logit = _logit(self._prior_rate)
        X, y = [], []
        for n, s in enumerate(past):
            pl = _logit(circuit_prior(flags[:n], s.circuit_key, g))  # only the races before this one
            for r in s.extra["weather"]["samples"]:
                f = {"rain_now": r[0], "since_rain_min": r[1], "run_min": r[2], "rc_risk": r[3], "rc_known": r[4],
                     "hum": r[5], "hum_tr": r[6], "trk_tr": r[7], "air_tr": r[8], "rain_seen": r[9]}
                X.append(design(f, pl))
                y.append(r[10])
        y = np.asarray(y)
        if y.sum() < MIN_TRAIN_POSITIVES or y.sum() == len(y):
            return
        from sklearn.linear_model import LogisticRegression

        self._model = LogisticRegression(C=1.0, max_iter=500).fit(np.asarray(X, float), y)

    def _prob(self, f: dict[str, float]) -> float:
        if not self._fitted:
            self._fit()
        if self._model is None:  # no history yet: persistence, plus the FIA figure
            if f["rain_now"]:
                return 0.9
            return min(0.02 + 0.3 * f["rc_risk"] * f["rc_known"] + 0.1 * math.exp(-max(f["since_rain_min"], 0) / 20.0) * f["rain_seen"], 0.6)
        return float(self._model.predict_proba(np.asarray([design(f, self._prior_logit)], float))[0, 1])

    # ------------------------------------------------------------------ following the race
    def observe(self, state):
        w = state.weather
        if w:
            self.series.add(state.t, tuple(w.get(k) for k in _FIELDS))

    def _rain(self, state, view):
        risk = view.race("rules").get("rc_rain_risk")
        key = (state.t, risk, len(self.series.t))
        if self._p_cache[0] != key:
            f = series_features(self.series, state.t, risk)
            self._p_cache = (key, (f, self._prob(f)))
        return self._p_cache[1]

    # ------------------------------------------------------------------ crossover
    def _crossover(self, state) -> dict:
        key = (state.t, len(state.laps))
        if self._cross_key == key:
            return self._cross
        self._cross_key = key
        cur = state.current_lap
        lo = cur - WINDOW_LAPS
        by_lap: dict[int, dict[str, list[float]]] = {}
        misses = 0
        for rec in reversed(state.laps):
            if rec.lap < lo:
                misses += 1
                if misses > 60:
                    break
                continue
            c = _compound_class(rec.compound)
            if c is None or rec.lap_time is None or rec.lap <= 1 or rec.is_in_lap or rec.is_out_lap or rec.track_status != "1":
                continue
            by_lap.setdefault(rec.lap, {"wet": [], "dry": []})[c].append(rec.lap_time)
        deltas = []
        for lap in sorted(by_lap):
            g = by_lap[lap]
            if g["wet"] and g["dry"]:
                deltas.append((lap, median(g["wet"]) - median(g["dry"])))
        c = self._cross
        if len(deltas) >= CROSS_MIN_LAPS or (deltas and abs(deltas[-1][1]) >= CROSS_ONE_LAP_S):
            d = mean([x for _, x in deltas[-2:]])
            c["delta"], c["seen_lap"] = d, cur
            call = "to_inters" if d <= -CROSS_MARGIN_S else "to_slicks" if d >= CROSS_MARGIN_S else "none"
            if call != "none" and call != c["call"]:
                c["call"], c["t"] = call, state.t
            elif call == "none" and abs(d) < CROSS_MARGIN_S / 2:
                c["call"], c["t"] = "none", None
        elif cur - c["seen_lap"] > CROSS_HOLD_LAPS:
            c["call"], c["t"], c["delta"] = "none", None, None
        return c

    # ------------------------------------------------------------------ values
    def race(self, state, view):
        w = state.weather
        f, p = self._rain(state, view)
        c = self._crossover(state)
        cars = [d for d in state.drivers.values() if d.running]
        n_wet = sum(d.compound in _WET for d in cars)
        n_dry = sum(d.compound in DRY_COMPOUNDS for d in cars)
        risk = view.race("rules").get("rc_rain_risk")
        return {
            "rainfall": w.get("Rainfall"),
            "track_temp": w.get("TrackTemp"),
            "air_temp": w.get("AirTemp"),
            "humidity": w.get("Humidity"),
            "wet_running": n_wet > 0,
            "rain_prob_10min": round(p, 4),
            "crossover": c["call"],
            "crossover_f": {"none": 0, "to_inters": 1, "to_slicks": -1}[c["call"]],
            "crossover_since": None if c["t"] is None else round(c["t"], 1),
            "inters_vs_slicks_s": None if c["call"] == "none" or c["delta"] is None else round(c["delta"], 2),
            "inters_vs_slicks_f": 0.0 if c["call"] == "none" or c["delta"] is None else round(c["delta"], 2),
            "rain_now": f["rain_now"],
            "rain_minutes": round(f["run_min"], 1),
            "minutes_since_rain": round(f["since_rain_min"], 1),
            "track_temp_trend": round(f["trk_tr"], 2),
            "air_temp_trend": round(f["air_tr"], 2),
            "humidity_trend": round(f["hum_tr"], 2),
            "rc_rain_risk_f": -1.0 if risk is None else float(risk),
            "prior_rain_rate": round(self._prior_rate, 4),
            "wet_cars": n_wet,
            "dry_cars": n_dry,
        }

    def alerts(self, state, view):
        v = view.race(self.name)
        out = []
        if v["rain_now"] == 0 and v["rain_prob_10min"] >= ALERT_PROB:
            out.append(Alert(state.t, self.name, "rain_onset_likely", "warn",
                             f"Rain likely within 10 min ({v['rain_prob_10min']:.0%}).",
                             data={"rain_prob_10min": v["rain_prob_10min"]}))
        if v["crossover"] != "none":
            tyre = "intermediates" if v["crossover"] == "to_inters" else "slicks"
            out.append(Alert(state.t, self.name, "crossover_reached", "warn",
                             f"Crossover reached: {tyre} are now the faster tyre"
                             + (f" by {abs(v['inters_vs_slicks_s']):.1f} s/lap." if v["inters_vs_slicks_s"] is not None else "."),
                             since=v["crossover_since"], data={"crossover": v["crossover"]}))
        return out

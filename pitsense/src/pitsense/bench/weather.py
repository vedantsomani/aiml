"""Weather benchmark (owner: weather): rain nowcast and slick / intermediate crossover.

Labels read the finished race (the future); they are only used to score.

* ``y_rain_10min``       lap-end rows: 1 if the feed's ``Rainfall`` flag is up at any moment in
                         [t, t + 10 min] (raining now counts), else 0.
* ``y_rain_onset_10min`` the same, but only on rows where it is dry now (NaN while raining).

Tasks (``rain_10min``) score ``weather__rain_prob_10min`` against baselines on all races, on
"wet races" (rain flag up at some lap end, or wet tyres used) and on the onset rows (dry now).

``python -m pitsense weather-report`` adds the per-race crossover table: the engineer's call
time against when switching paid off and when the median car switched.
"""

from __future__ import annotations

import warnings
from bisect import bisect_right
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

from ..pitwall.engineers.weather import HORIZON_S, _compound_class, _raining, _read_weather
from ..state import RaceState
from .metrics import CAL_BINS, binary_scores, calibration_table
from .models import BaseRate
from .tasks import Task, lap_end_rows


# ----------------------------------------------------------------------------- labels
def weather_labels(rows: list[dict], final: RaceState) -> None:
    s = _read_weather(final.meta)
    for row in rows:
        if s is None or not s.t:
            row["y_rain_10min"] = row["y_rain_onset_10min"] = float("nan")
            continue
        t = row["t"]
        i = s.index(t)
        j = bisect_right(s.t, t + HORIZON_S)
        now = i >= 0 and _raining(s.v[i])
        y = int(any(_raining(s.v[k]) for k in range(max(i, 0), j)))
        row["y_rain_10min"] = float(y)
        row["y_rain_onset_10min"] = float("nan") if now else float(y)


# ----------------------------------------------------------------------------- models
def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy(float) if name in df else np.full(len(df), default)


class Nowcast:
    """The engineer's ``rain_prob_10min``."""

    name = "nowcast"

    def fit(self, df, target):
        return self

    def predict(self, df):
        return _col(df, "weather__rain_prob_10min", 0.02)


class RainNow:
    """Persistence: the observed rain rate given the flag is up / down now (fit on earlier races)."""

    name = "persistence"

    def fit(self, df, target):
        now = _col(df, "weather__rain_now") > 0
        y = df[target].to_numpy(float)
        self.p = [float(y[~now].mean()) if (~now).any() else 0.02, float(y[now].mean()) if now.any() else 0.9]
        return self

    def predict(self, df):
        return np.where(_col(df, "weather__rain_now") > 0, self.p[1], self.p[0])


class RCRisk:
    """The FIA "risk of rain" figure taken at face value (2% before the first message)."""

    name = "rc_risk_raw"

    def fit(self, df, target):
        return self

    def predict(self, df):
        r = _col(df, "weather__rc_rain_risk_f", -1.0)
        return np.where(r < 0, 0.02, np.clip(r / 100.0, 0.01, 0.99))


class WeatherGBM:
    """Gradient boosting on the live weather columns, trained across earlier races."""

    name = "gbm_weather"
    cols = ("weather__rain_now", "weather__rain_minutes", "weather__minutes_since_rain", "weather__track_temp_trend",
            "weather__air_temp_trend", "weather__humidity_trend", "weather__rc_rain_risk_f", "weather__prior_rain_rate",
            "weather__humidity", "weather__track_temp", "weather__air_temp")

    def _x(self, df):
        return np.column_stack([_col(df, c, -1.0) for c in self.cols])

    def fit(self, df, target):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.m = HistGradientBoostingClassifier(max_depth=3, max_iter=80, learning_rate=0.1, random_state=0)
        self.m.fit(self._x(df), df[target].astype(int))
        return self

    def predict(self, df):
        return self.m.predict_proba(self._x(df))[:, 1]


MODELS = (BaseRate, RainNow, RCRisk, WeatherGBM, Nowcast)


# ----------------------------------------------------------------------------- task
def rain_rows(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["kind"] == "lap_end") & df["y_rain_10min"].notna()].copy()
    return d


def _wet_race_ids(df: pd.DataFrame) -> set:
    g = df.groupby("race_id").agg(rain=("weather__rainfall", "max"), wet=("weather__wet_running", "max"))
    return set(g[(g.rain.fillna(0) > 0) | (g.wet.fillna(0) > 0)].index)


def _subsets(P: pd.DataFrame) -> dict[str, pd.DataFrame]:
    wet = P[P.wet_race]
    return {"all": P, "wet_races": wet, "onset (dry now)": P[P.weather__rain_now == 0].dropna(subset=["y_rain_onset_10min"]),
            "onset, wet races": wet[wet.weather__rain_now == 0]}


def evaluate_rain(df: pd.DataFrame, *, test_year: int = 2026, min_train_races: int = 10):
    from dataclasses import replace

    from .evaluate import EvalResult, evaluate_task

    d = rain_rows(df)
    d["wet_race"] = d.race_id.isin(_wet_race_ids(d))
    base = Task("rain_10min", "binary", "y_rain_10min", rain_rows, MODELS,
                keep=("wet_race", "weather__rain_now", "y_rain_onset_10min"))
    res = evaluate_task(replace(base, select=lambda x: d[d.race_id.isin(x.race_id.unique())]), df, test_year, min_train_races)
    P = res.predictions
    names = [m.name for m in MODELS]
    rows, cal = [], {}
    for sub, S in _subsets(P).items():
        if not len(S):
            continue
        y = S["y_rain_10min"].to_numpy(float)
        for n in names:
            key = f"{n} [{sub}]"
            rows.append({"model": key, "subset": sub, "name": n, "races": S.race_id.nunique(), **binary_scores(y, S[n].to_numpy(float))})
            cal[key] = calibration_table(y, S[n].to_numpy(float), CAL_BINS)
    board = pd.DataFrame(rows)
    ref = board[board.name == "base_rate"].set_index("subset")["brier"]
    board["brier_skill"] = 1 - board["brier"] / board["subset"].map(ref)
    board = board.sort_values(["subset", "log_loss"]).reset_index(drop=True)
    return EvalResult("rain_10min", board, res.per_race, P, cal)


def weather_tasks(**_) -> list[Task]:
    return [Task("rain_10min", "binary", "y_rain_10min", rain_rows, MODELS,
                 keep=("weather__rain_now", "weather__rainfall", "weather__wet_running", "y_rain_onset_10min"),
                 evaluate=evaluate_rain)]


# ----------------------------------------------------------------------------- crossover table
def _lead_lap(final: RaceState, t: float) -> int:
    """Leader's lap count at session time t."""
    best = 0
    for x in final.laps:
        if x.t_end <= t and x.lap > best:
            best = x.lap
    return best


def switch_episodes(final: RaceState, gap_laps: int = 8, min_cars: int = 3) -> list[dict]:
    """Groups of cars moving the same way between slicks and wet tyres (red-flag stops left out)."""
    rec = {(x.driver, x.lap): x for x in final.laps}
    ev = {"to_inters": [], "to_slicks": []}
    for pe in final.pit_events:
        if pe.under_red:
            continue
        a, b = rec.get((pe.driver, pe.in_lap)), rec.get((pe.driver, pe.in_lap + 1))
        ca, cb = _compound_class(a.compound if a else None), _compound_class(b.compound if b else None)
        if ca == "dry" and cb == "wet":
            ev["to_inters"].append((pe.in_lap, pe.in_t, pe.driver))
        elif ca == "wet" and cb == "dry":
            ev["to_slicks"].append((pe.in_lap, pe.in_t, pe.driver))
    out = []
    for direction, items in ev.items():
        items.sort()
        groups, cur = [], []
        for it in items:
            if cur and it[0] - cur[-1][0] > gap_laps:
                groups.append(cur)
                cur = []
            cur.append(it)
        if cur:
            groups.append(cur)
        for k, g in enumerate(g for g in groups if len(g) >= min_cars):
            out.append({"direction": direction, "episode": k, "events": g})
    return out


def paid_off_lap(final: RaceState, ep: dict, k_first: int = 3) -> int | None:
    """First lap from which the earliest switchers are ahead of the field's typical progress.

    For each of the first ``k_first`` switchers: time elapsed since the lap before its stop, against the
    median of every other car over the same laps (their stops and tyres included). The switch has paid
    off once the switchers' median is below the field's on two laps in a row, the stop included.
    """
    rec = {(x.driver, x.lap): x for x in final.laps}
    drivers = sorted({d for d, _ in rec})
    G = ep["events"][:k_first]
    gset = {g[2] for g in G}
    last = max(x.lap for x in final.laps)
    gains: dict[int, list[float]] = {}
    for in_lap, _, drv in G:
        p = in_lap - 1
        if (drv, p) not in rec:
            continue
        for L in range(in_lap + 1, last + 1):
            if (drv, L) not in rec:
                break
            mine = rec[(drv, L)].t_end - rec[(drv, p)].t_end
            ref = [rec[(o, L)].t_end - rec[(o, p)].t_end for o in drivers
                   if o not in gset and (o, p) in rec and (o, L) in rec]
            if len(ref) >= 5:
                gains.setdefault(L, []).append(median(ref) - mine)
    start = max(g[0] for g in G) + 1
    prev = False
    for L in sorted(gains):
        if L < start or len(gains[L]) < len(G):
            continue
        ok = median(gains[L]) > 0
        if ok and prev:
            return L - 1
        prev = ok
    return None


def _call_runs(d: pd.DataFrame, direction: str) -> list[tuple[float, float]]:
    """Time spans (first row, last row) during which the engineer's call was ``direction``."""
    t = d.drop_duplicates("t").sort_values("t")
    runs, start, prev = [], None, None
    for tt, c in zip(t.t, t["weather__crossover"]):
        if c == direction and start is None:
            start = tt
        elif c != direction and start is not None:
            runs.append((start, prev))
            start = None
        prev = tt
    if start is not None:
        runs.append((start, prev))
    return runs


def crossover_table(df: pd.DataFrame, race_ids: list[str]) -> pd.DataFrame:
    from ..archive import downloaded_sessions
    from ..events import load_archive_session
    from ..state import replay

    refs = {r.slug: r for r in downloaded_sessions() if r.session_name == "Race"}
    out = []
    for rid in race_ids:
        final = replay(load_archive_session(refs[rid].local_dir))
        d = df[(df.race_id == rid) & (df.kind == "lap_end")]
        for ep in switch_episodes(final):
            ev = ep["events"]
            first_t = ev[0][1]
            # the engineer's call that is up around the switching: it may start before the first
            # switcher (cars that began on the other tyre) or after it
            last_t = ev[-1][1]
            runs = [r for r in _call_runs(d, ep["direction"]) if r[1] >= first_t - 600 and r[0] <= last_t + 600]
            call_t = float(runs[0][0]) if runs else None
            med_lap = float(median([e[0] for e in ev]))
            call_lap = _lead_lap(final, call_t) if call_t is not None else None
            other = "to_slicks" if ep["direction"] == "to_inters" else "to_inters"
            oruns = [r for r in _call_runs(d, other) if r[1] >= first_t - 600 and r[0] <= last_t + 600]
            opp_lap = _lead_lap(final, oruns[0][0]) if oruns else None
            po = paid_off_lap(final, ep)
            out.append({"race": rid, "direction": ep["direction"], "switchers": len(ev), "first_switch_lap": ev[0][0],
                        "call_lap": call_lap, "opposite_call_lap": opp_lap, "paid_off_lap": po, "median_switch_lap": med_lap,
                        "call_minus_first": None if call_lap is None else call_lap - ev[0][0],
                        "median_minus_call": None if call_lap is None else med_lap - call_lap})
    t = pd.DataFrame(out)
    for c in ("call_lap", "opposite_call_lap", "paid_off_lap", "call_minus_first", "median_minus_call"):
        if c in t:
            t[c] = pd.to_numeric(t[c], errors="coerce")
    return t


def false_calls(df: pd.DataFrame) -> pd.DataFrame:
    """Races where the engineer called a crossover but no episode of >= 3 cars switched that way."""
    d = df[(df.kind == "lap_end") & (df.weather__crossover != "none")]
    return d.groupby(["race_id", "weather__crossover"]).agg(rows=("t", "size"), first_t=("t", "min")).reset_index()


# ----------------------------------------------------------------------------- report
def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) or v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def cmd_weather_report(a) -> None:
    from .dataset import load_bench
    from .evaluate import evaluate_task

    warnings.filterwarnings("ignore")
    df = load_bench()
    text = []
    for year in a.test_years:
        res = evaluate_task(weather_tasks()[0], df, test_year=year, min_train_races=a.min_train)
        P = res.predictions
        wet = sorted(P[P.wet_race].race_id.unique()) if "wet_race" in P else []
        text.append(f"## rain_10min, test year {year}: {P.race_id.nunique()} races, {len(wet)} wet\n")
        b = res.leaderboard
        cols = ["subset", "name", "races", "n", "positives", "log_loss", "brier", "brier_skill", "auc"]
        text.append(_md(b[cols].round(4)) + "\n")
        for sub in ("all", "wet_races"):
            key = f"nowcast [{sub}]"
            if key in res.calibration:
                text.append(f"Calibration of nowcast ({sub}):\n\n" + _md(res.calibration[key].assign(bin=lambda x: x["bin"].astype(str)).round(4)) + "\n")
        # bootstrap over races: nowcast minus base-rate log loss on wet races
        W = P[P.wet_race]
        if W.race_id.nunique() >= 3:
            rng = np.random.default_rng(0)
            ids = sorted(W.race_id.unique())
            from .metrics import log_loss

            diffs = []
            for _ in range(500):
                pick = rng.choice(ids, len(ids))
                S = pd.concat([W[W.race_id == r] for r in pick])
                y = S.y_rain_10min.to_numpy(float)
                diffs.append(log_loss(y, S["nowcast"].to_numpy(float)) - log_loss(y, S["base_rate"].to_numpy(float)))
            text.append(f"Wet races, nowcast minus base-rate log loss, race bootstrap: median {np.median(diffs):.4f}, "
                        f"90% interval [{np.percentile(diffs, 5):.4f}, {np.percentile(diffs, 95):.4f}] ({len(ids)} races)\n")
    yrs = a.table_years
    d = df[df.year.isin(yrs)]
    wet = sorted(_wet_race_ids(d))
    tab = crossover_table(d, wet)
    text.append(f"## Crossover timing, wet races of {yrs} (laps are the leader's lap; episodes of >= 3 cars)\n")
    text.append(_md(tab) + "\n")
    if len(tab):
        ok = tab.dropna(subset=["call_lap"])
        text.append(f"Episodes {len(tab)}; called {len(ok)}; call before the median switcher: "
                    f"{int((ok.median_minus_call > 0).sum())}; before payoff: "
                    f"{int((ok.call_lap <= ok.paid_off_lap).sum())} of {int(ok.paid_off_lap.notna().sum())} with a payoff lap.\n")
    fc = false_calls(d)
    called = set(zip(tab.race, tab.direction)) if len(tab) else set()
    fc = fc[[(r, c) not in called for r, c in zip(fc.race_id, fc.weather__crossover)]]
    text.append(f"## Calls with no matching switch episode: {len(fc)}\n\n" + (_md(fc) if len(fc) else "") + "\n")
    out = "\n".join(text)
    print(out)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(out, encoding="utf-8")


def add_commands(sub) -> None:
    s = sub.add_parser("weather-report", help="weather engineer: rain nowcast scores and crossover timing")
    s.add_argument("--test-years", type=int, nargs="+", default=[2026])
    s.add_argument("--table-years", type=int, nargs="+", default=[2026])
    s.add_argument("--min-train", type=int, default=10)
    s.add_argument("--out", help="markdown file to write")
    s.set_defaults(fn=cmd_weather_report)
